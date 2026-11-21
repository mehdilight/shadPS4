// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(scope)),
                        ctx.ConstU32(static_cast<u32>(semantics)));
}
} // Anonymous namespace

void EmitBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory;
    spv::MemorySemanticsMask memory_semantics;
    if (ctx.sw_stage == Shader::SwStage::TessellationControl) {
        memory = spv::Scope::Invocation;
        memory_semantics = spv::MemorySemanticsMask::MaskNone;
    } else {
        memory = spv::Scope::Workgroup;
        memory_semantics =
            spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory;
    }
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

void EmitBalanceBarriers(EmitContext& ctx, Id cond, u32 count) {
    ctx.AddCapability(spv::Capability::GroupNonUniform);
    ctx.AddCapability(spv::Capability::GroupNonUniformVote);
    const Id subgroup_scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup))};
    const Id skipped{ctx.OpGroupNonUniformAll(ctx.U1[1], subgroup_scope, ctx.OpLogicalNot(ctx.U1[1], cond))};
    const Id balance_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(skipped, balance_label, merge_label);
    ctx.AddLabel(balance_label);
    for (u32 i = 0; i < count; ++i) {
        EmitBarrier(ctx);
    }
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitSubgroupBarrier(EmitContext& ctx) {
    const auto semantics{static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                          spv::MemorySemanticsMask::WorkgroupMemory)};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup)),
                        ctx.ConstU32(semantics));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

} // namespace Shader::Backend::SPIRV
