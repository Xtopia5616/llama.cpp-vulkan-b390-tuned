#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

#ifdef B390_TUNED

void load_row_ids(uint expert_idx, uint ic) {
    _ne1 = data_expert_count[expert_idx];
    const uint row_start = ic * BN;
    const uint row_count = row_start < _ne1 ? min(BN, _ne1 - row_start) : 0;
    const uint route_base = expert_idx * p.nei0 * p.nei1 + row_start;
    for (uint i = gl_LocalInvocationIndex; i < row_count; i += BLOCK_SIZE) {
        row_ids[i] = data_route[route_base + i];
    }
    barrier();
}

#else

#ifdef MUL_MAT_ID_USE_SUBGROUPS
shared uvec4 ballots_sh[NUM_WARPS];

void load_row_ids(uint expert_idx, bool nei0_is_pow2, uint ic) {
    _ne1 = 0;
    const uint num_elements = p.nei1 * p.nei0;
    const uint nei0shift = findLSB(p.nei0);
    uint ids[16];
    uint iter = 0;
    const uint expert_count = data_expert_count[expert_idx];

    for (uint j = 0; j < num_elements; j += BLOCK_SIZE) {
        if (iter == 0) {
            [[unroll]] for (uint k = 0; k < 16; ++k) {
                const uint i = j + gl_LocalInvocationIndex + k * BLOCK_SIZE;
                const bool in_range = i < num_elements;
                const uint ii1 = nei0_is_pow2 ? (i >> nei0shift) : (i / p.nei0);
                const uint ii0 = i - ii1 * p.nei0;
                ids[k] = in_range ? data_ids[ii1 * p.nbi1 + ii0] : 0;
            }
        }
        const uint i = j + gl_LocalInvocationIndex;
        const bool in_range = i < num_elements;
        const uint ii1 = nei0_is_pow2 ? (i >> nei0shift) : (i / p.nei0);
        const uint ii0 = i - ii1 * p.nei0;
        const uint id = ids[iter++];
        const uvec4 ballot = subgroupBallot(in_range && id == expert_idx);
        if (gl_SubgroupInvocationID == 0) ballots_sh[gl_SubgroupID] = ballot;
        barrier();

        uint subgroup_base = 0;
        uint total = 0;
        for (uint k = 0; k < gl_NumSubgroups; ++k) {
            if (k == gl_SubgroupID) subgroup_base = total;
            total += subgroupBallotBitCount(ballots_sh[k]);
        }
        barrier();

        const uint idx = subgroup_base + subgroupBallotExclusiveBitCount(ballot);
        if (in_range && id == expert_idx && _ne1 + idx >= ic * BN && _ne1 + idx < (ic + 1) * BN) {
            row_ids[_ne1 + idx - ic * BN] = u16vec2(ii0, ii1);
        }
        _ne1 += total;
        iter &= 15;
        if (_ne1 >= (ic + 1) * BN || _ne1 == expert_count) break;
    }
    barrier();
}
#endif // MUL_MAT_ID_USE_SUBGROUPS

void load_row_ids_hoisted(uint expert_idx, uint ic) {
    _ne1 = uint(data_expert_count[expert_idx]);
    const uint tile_begin = ic * BN;
    const uint tile_count = tile_begin < _ne1 ? min(BN, _ne1 - tile_begin) : 0;
    const uint expert_offset = uint(data_expert_count[p.n_experts + expert_idx]);
    const uint row_ids_offset = 2 * p.n_experts + 1 + expert_offset + tile_begin;
    for (uint i = gl_LocalInvocationIndex; i < tile_count; i += BLOCK_SIZE) {
        const uint packed_row_id = uint(data_expert_count[row_ids_offset + i]);
        row_ids[i] = u16vec2(packed_row_id & 0xffffu, packed_row_id >> 16);
    }
    barrier();
}

#endif // B390_TUNED
#endif // MUL_MAT_ID
