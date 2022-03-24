// This is my variant of AVX2 "generic" SIMD algorithm described in
// http://0x80.pl/articles/simd-strfind.html . Read two 32B lines at once and
// then or them together into a 8B bitmask, which speeds up the algorithm due to
// the SIMDs very low throughput. We also read the first 32B into a mask and
// check that against potential matches before going into the memcmp. It's
// possible to templatize this function and add handlers for smaller sizes etc.
// but I believe blowing up the code size would not be worth it.
//
// Implementation by Mateusz Dudzinski

static inline size_t
avx2_strstr_v3_simd(const char* s, size_t n, const char* needle, size_t k)
{
    int const pattern_is_short = k <= 32;

    // Load first 32 bytes of needle into an AVX reg. If needle is shorter than
    // 32 bytes the mask will allow us to go branchless later.
    uint8_t stack_needle[32] = {0};
    uint8_t stack_mask[32] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    };

    for (ptrdiff_t i = 0; i < (ptrdiff_t) k && i < 32; ++i)
        stack_needle[i] = needle[i];

    for (ptrdiff_t i = 0; i < (ptrdiff_t) k && i < 32; ++i)
        stack_mask[i] = 0;


    __m256i needle_v = _mm256_loadu_si256((__m256i*) stack_needle);
    __m256i needle_v_mask = _mm256_loadu_si256((__m256i*) stack_mask);
    __m256i first = _mm256_set1_epi8(needle[0]);
    __m256i last  = _mm256_set1_epi8(needle[k - 1]);

    for (ptrdiff_t i = 0; i < (ptrdiff_t) n; i += 64)
    {
        __m256i block_first_1 = _mm256_loadu_si256((__m256i*) (s + i));
        __m256i block_last_1  = _mm256_loadu_si256((__m256i*) (s + i + k - 1));
        __m256i eq_first_1 = _mm256_cmpeq_epi8(first, block_first_1);
        __m256i eq_last_1  = _mm256_cmpeq_epi8(last, block_last_1);
        uint32_t mask_1 = _mm256_movemask_epi8(_mm256_and_si256(eq_first_1, eq_last_1));

        __m256i block_first_2 = _mm256_loadu_si256((__m256i*) (s + i + 32));
        __m256i block_last_2  = _mm256_loadu_si256((__m256i*) (s + i + 32 + k - 1));
        __m256i eq_first_2 = _mm256_cmpeq_epi8(first, block_first_2);
        __m256i eq_last_2  = _mm256_cmpeq_epi8(last, block_last_2);
        uint32_t mask_2 = _mm256_movemask_epi8(_mm256_and_si256(eq_first_2, eq_last_2));

        uint64_t mask = ((uint64_t) mask_1) | (((uint64_t) mask_2) << 32);
        for (; mask; mask &= (mask - 1))
        {
            int32_t idx = __builtin_ctzll(mask);
            __m256i first_32b = _mm256_loadu_si256((__m256i*) (s + i + idx));
            __m256i first_32_cmp = _mm256_cmpeq_epi8(first_32b, needle_v);
            __m256i first_32_cmp_masked = _mm256_or_si256(first_32_cmp, needle_v_mask);

            // We need a match for the first k bytes (bytesfrom k to 31) have
            // already set bits in the movemask because of how we've set up
            // stack_mask
            if (_mm256_movemask_epi8(first_32_cmp_masked) == -1)
            {
                // If a pattern was shorter than 32B we are done. Otherwise we
                // need to memcmp the rest. Checking on pattern_is_short should
                // be fast and cheap and predicted ahead of time.
                if (pattern_is_short || memcmp(s + i + 32 + idx, needle + 32, k - 32) == 0)
                {
                    return i + idx;
                }
            }
        }
    }

    return std::string::npos;
}

extern size_t
avx2_strstr_v3(const char* s, size_t n, const char* needle, size_t k)
{
    if (__builtin_expect(n < k, 0))
        return std::string::npos;

    // This tries to handle correctly all the inputs (don't read past allocated
    // strings etc), which makes it (quite notably) slower. If you want to give
    // this function more edge during benchmarks, just tailcall to
    // avx2_strstr_v3_simd(s, n, needle, k)

    // Calculate the number of bytes of the s on which it is safe to use SIMD
    ptrdiff_t safe_n = 0;
    if (n > 2 * 64 + k)
    {
        ptrdiff_t safe_n = (n & (~(64 - 1))) - 64 - k + 1;
        size_t result = avx2_strstr_v3_simd(s, safe_n, needle, k);
        if (result != std::string::npos)
            return result;
    }

    // ... and check the tail manually
    for (ptrdiff_t i = safe_n; i < (ptrdiff_t) n - (ptrdiff_t) k + 1; ++i)
    {
        ptrdiff_t j = 0;
        for (; j < (ptrdiff_t) k; ++j)
            if (s[i + j] != needle[j])
                break;

        if (j == (ptrdiff_t) k)
            return i;
    }

    return std::string::npos;
}

extern size_t
avx2_strstr_v3(const std::string& s, const std::string& needle)
{
    return avx2_strstr_v3(s.data(), s.size(), needle.data(), needle.size());
}
