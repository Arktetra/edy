import flex_gemm

SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = "masked_implicit_gemm_splitk"
FLEX_GEMM_HASHMAP_RATIO = 2.0


# def __from_env():
#     import os

#     global SPCONV_ALGO
#     env_spconv_algo = os.environ.get("SPCONV_ALGO")
#     if env_spconv_algo is not None and env_spconv_algo in ["auto", "implicit_gemm", "native"]:
#         SPCONV_ALGO = env_spconv_algo
#     print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")


# __from_env()

flex_gemm.ops.spconv.set_algorithm(FLEX_GEMM_ALGO)
flex_gemm.ops.spconv.set_hashmap_ratio(FLEX_GEMM_HASHMAP_RATIO)

print(f"[Conv] using backend {FLEX_GEMM_ALGO}")
