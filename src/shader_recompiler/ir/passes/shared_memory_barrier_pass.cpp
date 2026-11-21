// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsWriteShared(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    if (opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>();
    }
    return opcode == IR::Opcode::WriteSharedU16 || opcode == IR::Opcode::WriteSharedU32 ||
           opcode == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
static void EmitBarrierInBlock(IR::Block* block, bool subgroup = false) {
    const auto emit_barrier = [subgroup](IR::IREmitter& ir) {
        if (subgroup) {
            ir.SubgroupBarrier();
        } else {
            ir.Barrier();
        }
    };
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                emit_barrier(ir);
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                emit_barrier(ir);
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        IR::IREmitter ir{*block, --block->end()};
        emit_barrier(ir);
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;
using BoundsCheckSet = std::unordered_set<const IR::AbstractSyntaxNode::Data*>;

static void EmitBarrierAtBlockStart(IR::Block* block) {
    auto insert_point = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
    IR::IREmitter ir{*block, insert_point};
    ir.Barrier();
}

static bool DependsOnInvocationId(const IR::Value& value, const RuntimeInfo& runtime_info) {
    if (value.IsImmediate()) {
        return false;
    }
    return IR::BreadthFirstSearch(value, [&runtime_info](IR::Inst* inst) -> std::optional<bool> {
               if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                   inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
                   const u32 component = inst->Arg(1).U32();
                   if (runtime_info.hw.cs.workgroup_size[component] > 1) {
                       return true;
                   }
               }
               return std::nullopt;
           }) == true;
}

static bool IsRangeComparison(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::SLessThan32:
    case IR::Opcode::ULessThan32:
    case IR::Opcode::SLessThanEqual32:
    case IR::Opcode::ULessThanEqual32:
    case IR::Opcode::SGreaterThan32:
    case IR::Opcode::UGreaterThan32:
    case IR::Opcode::SGreaterThanEqual32:
    case IR::Opcode::UGreaterThanEqual32:
        return true;
    default:
        return false;
    }
}

static bool IsGlobalCoordinate(const IR::Value& value) {
    return IR::BreadthFirstSearch(value, [](IR::Inst* inst) -> std::optional<bool> {
               if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                   inst->Arg(0).Attribute() == IR::Attribute::WorkgroupId) {
                   return true;
               }
               return std::nullopt;
           }) == true;
}

static bool IsDispatchConstant(const IR::Value& value) {
    if (value.IsImmediate()) {
        return false;
    }
    const IR::Inst* inst = value.Inst();
    switch (inst->GetOpcode()) {
    case IR::Opcode::ReadConst:
    case IR::Opcode::GetUserData:
        return true;
    default:
        return false;
    }
}

// Comparing an invocation coordinate against a runtime dimension only diverges along the edge of
// the dispatch, every other workgroup takes the branch as a whole.
static bool IsBoundsCheck(const IR::Value& cond, const RuntimeInfo& runtime_info) {
    if (cond.IsImmediate()) {
        return true;
    }
    const IR::Inst* inst = cond.Inst();
    const IR::Opcode opcode = inst->GetOpcode();
    if (opcode == IR::Opcode::ConditionRef) {
        return IsBoundsCheck(inst->Arg(0), runtime_info);
    }
    if (opcode == IR::Opcode::LogicalNot && !inst->Arg(0).IsImmediate() &&
        inst->Arg(0).Inst()->GetOpcode() == IR::Opcode::LogicalNot) {
        return IsBoundsCheck(inst->Arg(0).Inst()->Arg(0), runtime_info);
    }
    if (opcode == IR::Opcode::LogicalAnd) {
        return IsBoundsCheck(inst->Arg(0), runtime_info) &&
               IsBoundsCheck(inst->Arg(1), runtime_info);
    }
    if (!IsRangeComparison(opcode)) {
        return !DependsOnInvocationId(cond, runtime_info);
    }
    const IR::Value lhs = inst->Arg(0);
    const IR::Value rhs = inst->Arg(1);
    const bool lhs_lane = DependsOnInvocationId(lhs, runtime_info);
    const bool rhs_lane = DependsOnInvocationId(rhs, runtime_info);
    if (lhs_lane == rhs_lane) {
        return !lhs_lane;
    }
    const IR::Value& coord = lhs_lane ? lhs : rhs;
    const IR::Value& bound = lhs_lane ? rhs : lhs;
    return IsGlobalCoordinate(coord) && IsDispatchConstant(bound);
}

static bool IsDivergent(const IR::U1& cond, const RuntimeInfo& runtime_info) {
    return DependsOnInvocationId(cond, runtime_info);
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth,
                                    const RuntimeInfo& runtime_info,
                                    const BoundsCheckSet& bounds_checks) {
    const IR::U1 cond = data.if_node.cond;
    if (IsDivergent(cond, runtime_info) && !bounds_checks.contains(&data)) {
        if (divergence_depth == 0) {
            EmitBarrierAtBlockStart(data.if_node.merge);
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
}

// A barrier inside a loop is invalid when different invocations leave on different iterations.
// Mark such loops so their shared-memory synchronization can be deferred to the merge block.
static NodeSet FindDivergentLoops(const IR::AbstractSyntaxList& syntax_list,
                                  const RuntimeInfo& runtime_info) {
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (IsDivergent(node.data.repeat.cond, runtime_info)) {
                divergent_loops.emplace(node.data.repeat.merge);
            }
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (IsDivergent(node.data.break_node.cond, runtime_info)) {
                divergent_loops.emplace(node.data.break_node.merge);
            }
            break;
        default:
            break;
        }
    }
    return divergent_loops;
}

static bool IsSharedMemoryAccess(const IR::Inst& inst) {
    return IsLoadShared(inst) || IsWriteShared(inst);
}

// A branch on a bounds check may be skipped by an entire host subgroup at the edge of the
// dispatch. It can only hold workgroup barriers when that subgroup can execute the same number of
// barriers instead, so loops touching shared memory and early exits are not allowed in its body.
static BoundsCheckSet FindBoundsChecks(const IR::AbstractSyntaxList& syntax_list,
                                       const RuntimeInfo& runtime_info) {
    using Type = IR::AbstractSyntaxNode::Type;
    BoundsCheckSet bounds_checks;
    for (auto it = syntax_list.begin(); it != syntax_list.end(); ++it) {
        if (it->type != Type::If || !IsBoundsCheck(it->data.if_node.cond, runtime_info)) {
            continue;
        }
        const IR::Block* merge = it->data.if_node.merge;
        u32 loop_depth{};
        bool valid = true;
        for (auto node = std::next(it); node != syntax_list.end(); ++node) {
            if (node->type == Type::EndIf && node->data.end_if.merge == merge) {
                break;
            }
            if (node->type == Type::Loop) {
                ++loop_depth;
            } else if (node->type == Type::Repeat) {
                --loop_depth;
            } else if (node->type == Type::Return || node->type == Type::Unreachable ||
                       node->type == Type::Break) {
                valid &= loop_depth > 0 && node->type == Type::Break;
            } else if (node->type == Type::Block && loop_depth > 0) {
                valid &= std::ranges::none_of(node->data.block->Instructions(),
                                              IsSharedMemoryAccess);
            }
        }
        if (valid) {
            bounds_checks.insert(&it->data);
        }
    }
    return bounds_checks;
}

static void BalanceBoundsCheckBarriers(const IR::AbstractSyntaxList& syntax_list,
                                       const BoundsCheckSet& bounds_checks) {
    using Type = IR::AbstractSyntaxNode::Type;
    for (auto it = syntax_list.begin(); it != syntax_list.end(); ++it) {
        if (it->type != Type::If || !bounds_checks.contains(&it->data)) {
            continue;
        }
        IR::Block* merge = it->data.if_node.merge;
        u32 num_barriers{};
        for (auto node = std::next(it); node != syntax_list.end(); ++node) {
            if (node->type == Type::EndIf && node->data.end_if.merge == merge) {
                break;
            }
            if (node->type == Type::Block) {
                num_barriers += static_cast<u32>(std::ranges::count_if(
                    node->data.block->Instructions(), [](const IR::Inst& inst) {
                        return inst.GetOpcode() == IR::Opcode::Barrier;
                    }));
            }
        }
        if (num_barriers == 0) {
            continue;
        }
        auto insert_point = std::ranges::find_if_not(merge->Instructions(), IR::IsPhi);
        IR::IREmitter ir{*merge, insert_point};
        ir.BalanceBarriers(it->data.if_node.cond, ir.Imm32(num_barriers));
    }
}

static constexpr u32 GcnSubgroupSize = 64;

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    if (shared_memory_size == 0) {
        return;
    }
    // The compiler can only omit barriers when the local workgroup size is the same as the HW
    // subgroup.
    if (threadgroup_size != GcnSubgroupSize || !profile.needs_lds_barriers) {
        // GCN waves can use shared memory between invocations without explicit barriers because
        // they run in lockstep. Host compilers can reorder these accesses, so keep their order
        // with subgroup memory barriers.
        for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
            if (node.type == IR::AbstractSyntaxNode::Type::Block) {
                EmitBarrierInBlock(node.data.block, true);
            }
        }
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program.syntax_list, runtime_info);
    const BoundsCheckSet bounds_checks = FindBoundsChecks(program.syntax_list, runtime_info);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth, runtime_info,
                                    bounds_checks);
            continue;
        }
        if (node.type == Type::Loop && divergent_loops.contains(node.data.loop.merge)) {
            ++divergence_depth;
            continue;
        }
        if (node.type == Type::Repeat && divergent_loops.contains(node.data.repeat.merge)) {
            ASSERT(divergence_depth > 0);
            --divergence_depth;
            if (divergence_depth == 0) {
                EmitBarrierAtBlockStart(node.data.repeat.merge);
            }
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0) {
            EmitBarrierInBlock(node.data.block);
        }
    }
    BalanceBoundsCheckBarriers(program.syntax_list, bounds_checks);
}

} // namespace Shader::Optimization
