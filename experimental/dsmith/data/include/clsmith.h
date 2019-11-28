#define safe_mul_hi_func_int8_t_s_s(_t1, _si1, _si2)                                                             \
  ({                                                                                                             \
    _t1 si1 = (_si1);                                                                                            \
    _t1 si2 = (_si2);                                                                                            \
  ((((si1) > ((int8_t)0)) && ((si2) > ((int8_t)0)) && ((si1) > ((INT8_MAX) / (si2)))) || \
  (((si1) > ((int8_t)0)) && ((si2) <= ((int8_t)0)) && ((si2) < ((INT8_MIN) / (si1)))) || \
  (((si1) <= ((int8_t)0)) && ((si2) > ((int8_t)0)) && ((si1) < ((INT8_MIN) / (si2)))) || \
  (((si1) <= ((int8_t)0)) && ((si2) <= ((int8_t)0)) && (si1) != ((int8_t)0)) && ((si2) < ((INT8_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));                                                                                        \
  })

#define safe_mad_hi_func_int8_t_s_s_s(_t1, _si1, _si2, _si3) \
  ({                                                         \
    _t1 si1 = (_si1);                                        \
    _t1 si2 = (_si2);                                        \
    _t1 si3 = (_si3);                                        \
    ({                                                       \
      _t1 tmp = mul_hi((si1), (si2));                        \
      ((((tmp) > ((int8_t)0)) && ((si3) > ((int8_t)0)) &&    \
        ((tmp) > (((int8_t)(INT8_MAX)) - (si3)))) ||         \
       (((tmp) < ((int8_t)0)) && ((si3) < ((int8_t)0)) &&    \
        ((tmp) < (((int8_t)(INT8_MIN)) - (si3)))))           \
          ? (si1)                                            \
          : mad_hi((si1), (si2), (si3));                     \
    });                                                      \
  })

#define safe_mul_hi_func_int16_t_s_s(_t1, _si1, _si2)                                                                \
  ({                                                                                                                 \
    _t1 si1 = (_si1);                                                                                                \
    _t1 si2 = (_si2);                                                                                                \
  ((((si1) > ((int16_t)0)) && ((si2) > ((int16_t)0)) && ((si1) > ((INT16_MAX) / (si2)))) || \
  (((si1) > ((int16_t)0)) && ((si2) <= ((int16_t)0)) && ((si2) < ((INT16_MIN) / (si1)))) || \
  (((si1) <= ((int16_t)0)) && ((si2) > ((int16_t)0)) && ((si1) < ((INT16_MIN) / (si2)))) || \
  (((si1) <= ((int16_t)0)) && ((si2) <= ((int16_t)0)) && (si1) != ((int16_t)0)) && ((si2) < ((INT16_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));                                                                                            \
  })

#define safe_mad_hi_func_int16_t_s_s_s(_t1, _si1, _si2, _si3) \
  ({                                                          \
    _t1 si1 = (_si1);                                         \
    _t1 si2 = (_si2);                                         \
    _t1 si3 = (_si3);                                         \
    ({                                                        \
      _t1 tmp = mul_hi((si1), (si2));                         \
      ((((tmp) > ((int16_t)0)) && ((si3) > ((int16_t)0)) &&   \
        ((tmp) > (((int16_t)(INT16_MAX)) - (si3)))) ||        \
       (((tmp) < ((int16_t)0)) && ((si3) < ((int16_t)0)) &&   \
        ((tmp) < (((int16_t)(INT16_MIN)) - (si3)))))          \
          ? (si1)                                             \
          : mad_hi((si1), (si2), (si3));                      \
    });                                                       \
  })

#define safe_mul_hi_func_int32_t_s_s(_t1, _si1, _si2)                                                                \
  ({                                                                                                                 \
    _t1 si1 = (_si1);                                                                                                \
    _t1 si2 = (_si2);                                                                                                \
  ((((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) > ((INT32_MAX) / (si2)))) || \
  (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si2) < ((INT32_MIN) / (si1)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) < ((INT32_MIN) / (si2)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) && (si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));                                                                                            \
  })

#define safe_mad_hi_func_int32_t_s_s_s(_t1, _si1, _si2, _si3) \
  ({                                                          \
    _t1 si1 = (_si1);                                         \
    _t1 si2 = (_si2);                                         \
    _t1 si3 = (_si3);                                         \
    ({                                                        \
      _t1 tmp = mul_hi((si1), (si2));                         \
      ((((tmp) > ((int32_t)0)) && ((si3) > ((int32_t)0)) &&   \
        ((tmp) > (((int32_t)(INT32_MAX)) - (si3)))) ||        \
       (((tmp) < ((int32_t)0)) && ((si3) < ((int32_t)0)) &&   \
        ((tmp) < (((int32_t)(INT32_MIN)) - (si3)))))          \
          ? (si1)                                             \
          : mad_hi((si1), (si2), (si3));                      \
    });                                                       \
  })

#define safe_mul_hi_func_int64_t_s_s(_t1, _si1, _si2)                                                                \
  ({                                                                                                                 \
    _t1 si1 = (_si1);                                                                                                \
    _t1 si2 = (_si2);                                                                                                \
  ((((si1) > ((int64_t)0)) && ((si2) > ((int64_t)0)) && ((si1) > ((INT64_MAX) / (si2)))) || \
  (((si1) > ((int64_t)0)) && ((si2) <= ((int64_t)0)) && ((si2) < ((INT64_MIN) / (si1)))) || \
  (((si1) <= ((int64_t)0)) && ((si2) > ((int64_t)0)) && ((si1) < ((INT64_MIN) / (si2)))) || \
  (((si1) <= ((int64_t)0)) && ((si2) <= ((int64_t)0)) && (si1) != ((int64_t)0)) && ((si2) < ((INT64_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));                                                                                            \
  })

#define safe_mad_hi_func_int64_t_s_s_s(_t1, _si1, _si2, _si3) \
  ({                                                          \
    _t1 si1 = (_si1);                                         \
    _t1 si2 = (_si2);                                         \
    _t1 si3 = (_si3);                                         \
    ({                                                        \
      _t1 tmp = mul_hi((si1), (si2));                         \
      ((((tmp) > ((int64_t)0)) && ((si3) > ((int64_t)0)) &&   \
        ((tmp) > (((int64_t)(INT64_MAX)) - (si3)))) ||        \
       (((tmp) < ((int64_t)0)) && ((si3) < ((int64_t)0)) &&   \
        ((tmp) < (((int64_t)(INT64_MIN)) - (si3)))))          \
          ? (si1)                                             \
          : mad_hi((si1), (si2), (si3));                      \
    });                                                       \
  })

#define safe_mul24_func_int32_t_s_s(_t1, _si1, _si2)               \
  ({                                                               \
    _t1 si1 = (_si1);                                              \
    _t1 si2 = (_si2);                                              \
    ((((si1) < (-(1 << 23))) || ((si1) > ((1 << 23) - 1)) ||       \
      ((si2) < (-(1 << 23))) || ((si2) > ((1 << 23) - 1))) ||      \
     (((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) &&          \
      ((si1) > ((INT32_MAX) / (si2)))) ||                          \
     (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) &&         \
      ((si2) < ((INT32_MIN) / (si1)))) ||                          \
     (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) &&         \
      ((si1) < ((INT32_MIN) / (si2)))) ||                          \
     (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) &&        \
      ((si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
        ? (si1)                                                    \
        : mul24((si1), (si2));                                     \
  })

#define safe_mad24_func_int32_t_s_s_s(_t1, _si1, _si2, _si3)       \
  ({                                                               \
    _t1 si1 = (_si1);                                              \
    _t1 si2 = (_si2);                                              \
    _t1 si3 = (_si3);                                              \
    ((((si1) < (-(1 << 23))) || ((si1) > ((1 << 23) - 1)) ||       \
      ((si2) < (-(1 << 23))) || ((si2) > ((1 << 23) - 1))) ||      \
     (((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) &&          \
      ((si1) > ((INT32_MAX) / (si2)))) ||                          \
     (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) &&         \
      ((si2) < ((INT32_MIN) / (si1)))) ||                          \
     (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) &&         \
      ((si1) < ((INT32_MIN) / (si2)))) ||                          \
     (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) &&        \
      ((si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
        ? (si1)                                                    \
        : ({                                                       \
            _t1 tmp = mul24((si1), (si2));                         \
            ((((tmp) > ((int32_t)0)) && ((si3) > ((int32_t)0)) &&  \
              ((tmp) > ((INT32_MAX) - (si3)))) ||                  \
             (((tmp) < ((int32_t)0)) && ((si3) < ((int32_t)0)) &&  \
              ((tmp) < ((INT32_MIN) - (si3)))))                    \
                ? (si1)                                            \
                : mad24((si1), (si2), (si3));                      \
          });                                                      \
  })

#define safe_mul24_func_uint32_t_u_u(_t1, _ui1, _ui2)         \
  ({                                                          \
    _t1 ui1 = (_ui1);                                         \
    _t1 ui2 = (_ui2);                                         \
    (((ui1) < (0)) || ((ui1) > (1 << 24)) || ((ui2) < (0)) || \
     ((ui2) > (1 << 24)))                                     \
        ? (ui1)                                               \
        : mul24((ui1), (ui2));                                \
  })

#define safe_mad24_func_uint32_t_u_u_u(_t1, _ui1, _ui2, _ui3) \
  ({                                                          \
    _t1 ui1 = (_ui1);                                         \
    _t1 ui2 = (_ui2);                                         \
    _t1 ui3 = (_ui3);                                         \
    (((ui1) < (0)) || ((ui1) > (1 << 24)) || ((ui2) < (0)) || \
     ((ui2) > (1 << 24)))                                     \
        ? (ui1)                                               \
        : mad24((ui1), (ui2), (ui3));                         \
  })

#define safe_clamp_func(_t1, _t2, _x, _y, _z) \
  ({                                          \
    _t1 x = (_x);                             \
    _t2 y = (_y);                             \
    _t2 z = (_z);                             \
    ((y) > (z)) ? (x) : clamp((x), (y), (z)); \
  })

#define safe_unary_minus_func_int8_t_s(_si)                              \
  ({                                                                     \
    int8_t si = (_si);                                                   \
    (((int8_t)(si)) == (INT8_MIN)) ? ((int8_t)(si)) : (-((int8_t)(si))); \
  })

#define safe_add_func_int8_t_s_s(_si1, _si2)                                \
  ({                                                                        \
    int8_t si1 = (_si1);                                                    \
    int8_t si2 = (_si2);                                                    \
    (((((int8_t)(si1)) > ((int8_t)0)) && (((int8_t)(si2)) > ((int8_t)0)) && \
      (((int8_t)(si1)) > ((INT8_MAX) - ((int8_t)(si2))))) ||                \
     ((((int8_t)(si1)) < ((int8_t)0)) && (((int8_t)(si2)) < ((int8_t)0)) && \
      (((int8_t)(si1)) < ((INT8_MIN) - ((int8_t)(si2))))))                  \
        ? ((int8_t)(si1))                                                   \
        : (((int8_t)(si1)) + ((int8_t)(si2)));                              \
  })

#define safe_sub_func_int8_t_s_s(_si1, _si2)                   \
  ({                                                           \
    int8_t si1 = (_si1);                                       \
    int8_t si2 = (_si2);                                       \
    (((((int8_t)(si1)) ^ ((int8_t)(si2))) &                    \
      (((((int8_t)(si1)) ^                                     \
         ((((int8_t)(si1)) ^ ((int8_t)(si2))) &                \
          (((int8_t)1) << (sizeof(int8_t) * CHAR_BIT - 1)))) - \
        ((int8_t)(si2))) ^                                     \
       ((int8_t)(si2)))) < ((int8_t)0))                        \
        ? ((int8_t)(si1))                                      \
        : (((int8_t)(si1)) - ((int8_t)(si2)));                 \
  })

#define safe_mul_func_int8_t_s_s(_si1, _si2)                                  \
  ({                                                                          \
    int8_t si1 = (_si1);                                                      \
    int8_t si2 = (_si2);                                                      \
    (((((int8_t)(si1)) > ((int8_t)0)) && (((int8_t)(si2)) > ((int8_t)0)) &&   \
      (((int8_t)(si1)) > ((INT8_MAX) / ((int8_t)(si2))))) ||                  \
     ((((int8_t)(si1)) > ((int8_t)0)) && (((int8_t)(si2)) <= ((int8_t)0)) &&  \
      (((int8_t)(si2)) < ((INT8_MIN) / ((int8_t)(si1))))) ||                  \
     ((((int8_t)(si1)) <= ((int8_t)0)) && (((int8_t)(si2)) > ((int8_t)0)) &&  \
      (((int8_t)(si1)) < ((INT8_MIN) / ((int8_t)(si2))))) ||                  \
     ((((int8_t)(si1)) <= ((int8_t)0)) && (((int8_t)(si2)) <= ((int8_t)0)) && \
      (((int8_t)(si1)) != ((int8_t)0)) &&                                     \
      (((int8_t)(si2)) < ((INT8_MAX) / ((int8_t)(si1))))))                    \
        ? ((int8_t)(si1))                                                     \
        : ((int8_t)(si1)) * ((int8_t)(si2));                                  \
  })

#define safe_mod_func_int8_t_s_s(_si1, _si2)                                 \
  ({                                                                         \
    int8_t si1 = (_si1);                                                     \
    int8_t si2 = (_si2);                                                     \
    ((((int8_t)(si2)) == ((int8_t)0)) ||                                     \
     ((((int8_t)(si1)) == (INT8_MIN)) && (((int8_t)(si2)) == ((int8_t)-1)))) \
        ? ((int8_t)(si1))                                                    \
        : (((int8_t)(si1)) % ((int8_t)(si2)));                               \
  })

#define safe_div_func_int8_t_s_s(_si1, _si2)                                 \
  ({                                                                         \
    int8_t si1 = (_si1);                                                     \
    int8_t si2 = (_si2);                                                     \
    ((((int8_t)(si2)) == ((int8_t)0)) ||                                     \
     ((((int8_t)(si1)) == (INT8_MIN)) && (((int8_t)(si2)) == ((int8_t)-1)))) \
        ? ((int8_t)(si1))                                                    \
        : (((int8_t)(si1)) / ((int8_t)(si2)));                               \
  })

#define safe_lshift_func_int8_t_s_s(_left, _right)                         \
  ({                                                                       \
    int8_t left = (_left);                                                 \
    int right = (_right);                                                  \
    ((((int8_t)(left)) < ((int8_t)0)) || (((int)(right)) < ((int8_t)0)) || \
     (((int)(right)) >= sizeof(int8_t) * CHAR_BIT) ||                      \
     (((int8_t)(left)) > ((INT8_MAX) >> ((int)(right)))))                  \
        ? ((int8_t)(left))                                                 \
        : (((int8_t)(left)) << ((int)(right)));                            \
  })

#define safe_lshift_func_int8_t_s_u(_left, _right)                 \
  ({                                                               \
    int8_t left = (_left);                                         \
    unsigned int right = (_right);                                 \
    ((((int8_t)(left)) < ((int8_t)0)) ||                           \
     (((unsigned int)(right)) >= sizeof(int8_t) * CHAR_BIT) ||     \
     (((int8_t)(left)) > ((INT8_MAX) >> ((unsigned int)(right))))) \
        ? ((int8_t)(left))                                         \
        : (((int8_t)(left)) << ((unsigned int)(right)));           \
  })

#define safe_rshift_func_int8_t_s_s(_left, _right)                         \
  ({                                                                       \
    int8_t left = (_left);                                                 \
    int right = (_right);                                                  \
    ((((int8_t)(left)) < ((int8_t)0)) || (((int)(right)) < ((int8_t)0)) || \
     (((int)(right)) >= sizeof(int8_t) * CHAR_BIT))                        \
        ? ((int8_t)(left))                                                 \
        : (((int8_t)(left)) >> ((int)(right)));                            \
  })

#define safe_rshift_func_int8_t_s_u(_left, _right)           \
  ({                                                         \
    int8_t left = (_left);                                   \
    unsigned int right = (_right);                           \
    ((((int8_t)(left)) < ((int8_t)0)) ||                     \
     (((unsigned int)(right)) >= sizeof(int8_t) * CHAR_BIT)) \
        ? ((int8_t)(left))                                   \
        : (((int8_t)(left)) >> ((unsigned int)(right)));     \
  })

#define safe_unary_minus_func_int16_t_s(_si)                                 \
  ({                                                                         \
    int16_t si = (_si);                                                      \
    (((int16_t)(si)) == (INT16_MIN)) ? ((int16_t)(si)) : (-((int16_t)(si))); \
  })

#define safe_add_func_int16_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int16_t si1 = (_si1);                                       \
    int16_t si2 = (_si2);                                       \
    (((((int16_t)(si1)) > ((int16_t)0)) &&                      \
      (((int16_t)(si2)) > ((int16_t)0)) &&                      \
      (((int16_t)(si1)) > ((INT16_MAX) - ((int16_t)(si2))))) || \
     ((((int16_t)(si1)) < ((int16_t)0)) &&                      \
      (((int16_t)(si2)) < ((int16_t)0)) &&                      \
      (((int16_t)(si1)) < ((INT16_MIN) - ((int16_t)(si2))))))   \
        ? ((int16_t)(si1))                                      \
        : (((int16_t)(si1)) + ((int16_t)(si2)));                \
  })

#define safe_sub_func_int16_t_s_s(_si1, _si2)                    \
  ({                                                             \
    int16_t si1 = (_si1);                                        \
    int16_t si2 = (_si2);                                        \
    (((((int16_t)(si1)) ^ ((int16_t)(si2))) &                    \
      (((((int16_t)(si1)) ^                                      \
         ((((int16_t)(si1)) ^ ((int16_t)(si2))) &                \
          (((int16_t)1) << (sizeof(int16_t) * CHAR_BIT - 1)))) - \
        ((int16_t)(si2))) ^                                      \
       ((int16_t)(si2)))) < ((int16_t)0))                        \
        ? ((int16_t)(si1))                                       \
        : (((int16_t)(si1)) - ((int16_t)(si2)));                 \
  })

#define safe_mul_func_int16_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int16_t si1 = (_si1);                                       \
    int16_t si2 = (_si2);                                       \
    (((((int16_t)(si1)) > ((int16_t)0)) &&                      \
      (((int16_t)(si2)) > ((int16_t)0)) &&                      \
      (((int16_t)(si1)) > ((INT16_MAX) / ((int16_t)(si2))))) || \
     ((((int16_t)(si1)) > ((int16_t)0)) &&                      \
      (((int16_t)(si2)) <= ((int16_t)0)) &&                     \
      (((int16_t)(si2)) < ((INT16_MIN) / ((int16_t)(si1))))) || \
     ((((int16_t)(si1)) <= ((int16_t)0)) &&                     \
      (((int16_t)(si2)) > ((int16_t)0)) &&                      \
      (((int16_t)(si1)) < ((INT16_MIN) / ((int16_t)(si2))))) || \
     ((((int16_t)(si1)) <= ((int16_t)0)) &&                     \
      (((int16_t)(si2)) <= ((int16_t)0)) &&                     \
      (((int16_t)(si1)) != ((int16_t)0)) &&                     \
      (((int16_t)(si2)) < ((INT16_MAX) / ((int16_t)(si1))))))   \
        ? ((int16_t)(si1))                                      \
        : ((int16_t)(si1)) * ((int16_t)(si2));                  \
  })

#define safe_mod_func_int16_t_s_s(_si1, _si2)    \
  ({                                             \
    int16_t si1 = (_si1);                        \
    int16_t si2 = (_si2);                        \
    ((((int16_t)(si2)) == ((int16_t)0)) ||       \
     ((((int16_t)(si1)) == (INT16_MIN)) &&       \
      (((int16_t)(si2)) == ((int16_t)-1))))      \
        ? ((int16_t)(si1))                       \
        : (((int16_t)(si1)) % ((int16_t)(si2))); \
  })

#define safe_div_func_int16_t_s_s(_si1, _si2)    \
  ({                                             \
    int16_t si1 = (_si1);                        \
    int16_t si2 = (_si2);                        \
    ((((int16_t)(si2)) == ((int16_t)0)) ||       \
     ((((int16_t)(si1)) == (INT16_MIN)) &&       \
      (((int16_t)(si2)) == ((int16_t)-1))))      \
        ? ((int16_t)(si1))                       \
        : (((int16_t)(si1)) / ((int16_t)(si2))); \
  })

#define safe_lshift_func_int16_t_s_s(_left, _right)                           \
  ({                                                                          \
    int16_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int16_t)(left)) < ((int16_t)0)) || (((int)(right)) < ((int16_t)0)) || \
     (((int)(right)) >= sizeof(int16_t) * CHAR_BIT) ||                        \
     (((int16_t)(left)) > ((INT16_MAX) >> ((int)(right)))))                   \
        ? ((int16_t)(left))                                                   \
        : (((int16_t)(left)) << ((int)(right)));                              \
  })

#define safe_lshift_func_int16_t_s_u(_left, _right)                  \
  ({                                                                 \
    int16_t left = (_left);                                          \
    unsigned int right = (_right);                                   \
    ((((int16_t)(left)) < ((int16_t)0)) ||                           \
     (((unsigned int)(right)) >= sizeof(int16_t) * CHAR_BIT) ||      \
     (((int16_t)(left)) > ((INT16_MAX) >> ((unsigned int)(right))))) \
        ? ((int16_t)(left))                                          \
        : (((int16_t)(left)) << ((unsigned int)(right)));            \
  })

#define safe_rshift_func_int16_t_s_s(_left, _right)                           \
  ({                                                                          \
    int16_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int16_t)(left)) < ((int16_t)0)) || (((int)(right)) < ((int16_t)0)) || \
     (((int)(right)) >= sizeof(int16_t) * CHAR_BIT))                          \
        ? ((int16_t)(left))                                                   \
        : (((int16_t)(left)) >> ((int)(right)));                              \
  })

#define safe_rshift_func_int16_t_s_u(_left, _right)           \
  ({                                                          \
    int16_t left = (_left);                                   \
    unsigned int right = (_right);                            \
    ((((int16_t)(left)) < ((int16_t)0)) ||                    \
     (((unsigned int)(right)) >= sizeof(int16_t) * CHAR_BIT)) \
        ? ((int16_t)(left))                                   \
        : (((int16_t)(left)) >> ((unsigned int)(right)));     \
  })

#define safe_unary_minus_func_int32_t_s(_si)                                 \
  ({                                                                         \
    int32_t si = (_si);                                                      \
    (((int32_t)(si)) == (INT32_MIN)) ? ((int32_t)(si)) : (-((int32_t)(si))); \
  })

#define safe_add_func_int32_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int32_t si1 = (_si1);                                       \
    int32_t si2 = (_si2);                                       \
    (((((int32_t)(si1)) > ((int32_t)0)) &&                      \
      (((int32_t)(si2)) > ((int32_t)0)) &&                      \
      (((int32_t)(si1)) > ((INT32_MAX) - ((int32_t)(si2))))) || \
     ((((int32_t)(si1)) < ((int32_t)0)) &&                      \
      (((int32_t)(si2)) < ((int32_t)0)) &&                      \
      (((int32_t)(si1)) < ((INT32_MIN) - ((int32_t)(si2))))))   \
        ? ((int32_t)(si1))                                      \
        : (((int32_t)(si1)) + ((int32_t)(si2)));                \
  })

#define safe_sub_func_int32_t_s_s(_si1, _si2)                    \
  ({                                                             \
    int32_t si1 = (_si1);                                        \
    int32_t si2 = (_si2);                                        \
    (((((int32_t)(si1)) ^ ((int32_t)(si2))) &                    \
      (((((int32_t)(si1)) ^                                      \
         ((((int32_t)(si1)) ^ ((int32_t)(si2))) &                \
          (((int32_t)1) << (sizeof(int32_t) * CHAR_BIT - 1)))) - \
        ((int32_t)(si2))) ^                                      \
       ((int32_t)(si2)))) < ((int32_t)0))                        \
        ? ((int32_t)(si1))                                       \
        : (((int32_t)(si1)) - ((int32_t)(si2)));                 \
  })

#define safe_mul_func_int32_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int32_t si1 = (_si1);                                       \
    int32_t si2 = (_si2);                                       \
    (((((int32_t)(si1)) > ((int32_t)0)) &&                      \
      (((int32_t)(si2)) > ((int32_t)0)) &&                      \
      (((int32_t)(si1)) > ((INT32_MAX) / ((int32_t)(si2))))) || \
     ((((int32_t)(si1)) > ((int32_t)0)) &&                      \
      (((int32_t)(si2)) <= ((int32_t)0)) &&                     \
      (((int32_t)(si2)) < ((INT32_MIN) / ((int32_t)(si1))))) || \
     ((((int32_t)(si1)) <= ((int32_t)0)) &&                     \
      (((int32_t)(si2)) > ((int32_t)0)) &&                      \
      (((int32_t)(si1)) < ((INT32_MIN) / ((int32_t)(si2))))) || \
     ((((int32_t)(si1)) <= ((int32_t)0)) &&                     \
      (((int32_t)(si2)) <= ((int32_t)0)) &&                     \
      (((int32_t)(si1)) != ((int32_t)0)) &&                     \
      (((int32_t)(si2)) < ((INT32_MAX) / ((int32_t)(si1))))))   \
        ? ((int32_t)(si1))                                      \
        : ((int32_t)(si1)) * ((int32_t)(si2));                  \
  })

#define safe_mod_func_int32_t_s_s(_si1, _si2)    \
  ({                                             \
    int32_t si1 = (_si1);                        \
    int32_t si2 = (_si2);                        \
    ((((int32_t)(si2)) == ((int32_t)0)) ||       \
     ((((int32_t)(si1)) == (INT32_MIN)) &&       \
      (((int32_t)(si2)) == ((int32_t)-1))))      \
        ? ((int32_t)(si1))                       \
        : (((int32_t)(si1)) % ((int32_t)(si2))); \
  })

#define safe_div_func_int32_t_s_s(_si1, _si2)    \
  ({                                             \
    int32_t si1 = (_si1);                        \
    int32_t si2 = (_si2);                        \
    ((((int32_t)(si2)) == ((int32_t)0)) ||       \
     ((((int32_t)(si1)) == (INT32_MIN)) &&       \
      (((int32_t)(si2)) == ((int32_t)-1))))      \
        ? ((int32_t)(si1))                       \
        : (((int32_t)(si1)) / ((int32_t)(si2))); \
  })

#define safe_lshift_func_int32_t_s_s(_left, _right)                           \
  ({                                                                          \
    int32_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int32_t)(left)) < ((int32_t)0)) || (((int)(right)) < ((int32_t)0)) || \
     (((int)(right)) >= sizeof(int32_t) * CHAR_BIT) ||                        \
     (((int32_t)(left)) > ((INT32_MAX) >> ((int)(right)))))                   \
        ? ((int32_t)(left))                                                   \
        : (((int32_t)(left)) << ((int)(right)));                              \
  })

#define safe_lshift_func_int32_t_s_u(_left, _right)                  \
  ({                                                                 \
    int32_t left = (_left);                                          \
    unsigned int right = (_right);                                   \
    ((((int32_t)(left)) < ((int32_t)0)) ||                           \
     (((unsigned int)(right)) >= sizeof(int32_t) * CHAR_BIT) ||      \
     (((int32_t)(left)) > ((INT32_MAX) >> ((unsigned int)(right))))) \
        ? ((int32_t)(left))                                          \
        : (((int32_t)(left)) << ((unsigned int)(right)));            \
  })

#define safe_rshift_func_int32_t_s_s(_left, _right)                           \
  ({                                                                          \
    int32_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int32_t)(left)) < ((int32_t)0)) || (((int)(right)) < ((int32_t)0)) || \
     (((int)(right)) >= sizeof(int32_t) * CHAR_BIT))                          \
        ? ((int32_t)(left))                                                   \
        : (((int32_t)(left)) >> ((int)(right)));                              \
  })

#define safe_rshift_func_int32_t_s_u(_left, _right)           \
  ({                                                          \
    int32_t left = (_left);                                   \
    unsigned int right = (_right);                            \
    ((((int32_t)(left)) < ((int32_t)0)) ||                    \
     (((unsigned int)(right)) >= sizeof(int32_t) * CHAR_BIT)) \
        ? ((int32_t)(left))                                   \
        : (((int32_t)(left)) >> ((unsigned int)(right)));     \
  })

#define safe_unary_minus_func_int64_t_s(_si)                                 \
  ({                                                                         \
    int64_t si = (_si);                                                      \
    (((int64_t)(si)) == (INT64_MIN)) ? ((int64_t)(si)) : (-((int64_t)(si))); \
  })

#define safe_add_func_int64_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int64_t si1 = (_si1);                                       \
    int64_t si2 = (_si2);                                       \
    (((((int64_t)(si1)) > ((int64_t)0)) &&                      \
      (((int64_t)(si2)) > ((int64_t)0)) &&                      \
      (((int64_t)(si1)) > ((INT64_MAX) - ((int64_t)(si2))))) || \
     ((((int64_t)(si1)) < ((int64_t)0)) &&                      \
      (((int64_t)(si2)) < ((int64_t)0)) &&                      \
      (((int64_t)(si1)) < ((INT64_MIN) - ((int64_t)(si2))))))   \
        ? ((int64_t)(si1))                                      \
        : (((int64_t)(si1)) + ((int64_t)(si2)));                \
  })

#define safe_sub_func_int64_t_s_s(_si1, _si2)                    \
  ({                                                             \
    int64_t si1 = (_si1);                                        \
    int64_t si2 = (_si2);                                        \
    (((((int64_t)(si1)) ^ ((int64_t)(si2))) &                    \
      (((((int64_t)(si1)) ^                                      \
         ((((int64_t)(si1)) ^ ((int64_t)(si2))) &                \
          (((int64_t)1) << (sizeof(int64_t) * CHAR_BIT - 1)))) - \
        ((int64_t)(si2))) ^                                      \
       ((int64_t)(si2)))) < ((int64_t)0))                        \
        ? ((int64_t)(si1))                                       \
        : (((int64_t)(si1)) - ((int64_t)(si2)));                 \
  })

#define safe_mul_func_int64_t_s_s(_si1, _si2)                   \
  ({                                                            \
    int64_t si1 = (_si1);                                       \
    int64_t si2 = (_si2);                                       \
    (((((int64_t)(si1)) > ((int64_t)0)) &&                      \
      (((int64_t)(si2)) > ((int64_t)0)) &&                      \
      (((int64_t)(si1)) > ((INT64_MAX) / ((int64_t)(si2))))) || \
     ((((int64_t)(si1)) > ((int64_t)0)) &&                      \
      (((int64_t)(si2)) <= ((int64_t)0)) &&                     \
      (((int64_t)(si2)) < ((INT64_MIN) / ((int64_t)(si1))))) || \
     ((((int64_t)(si1)) <= ((int64_t)0)) &&                     \
      (((int64_t)(si2)) > ((int64_t)0)) &&                      \
      (((int64_t)(si1)) < ((INT64_MIN) / ((int64_t)(si2))))) || \
     ((((int64_t)(si1)) <= ((int64_t)0)) &&                     \
      (((int64_t)(si2)) <= ((int64_t)0)) &&                     \
      (((int64_t)(si1)) != ((int64_t)0)) &&                     \
      (((int64_t)(si2)) < ((INT64_MAX) / ((int64_t)(si1))))))   \
        ? ((int64_t)(si1))                                      \
        : ((int64_t)(si1)) * ((int64_t)(si2));                  \
  })

#define safe_mod_func_int64_t_s_s(_si1, _si2)    \
  ({                                             \
    int64_t si1 = (_si1);                        \
    int64_t si2 = (_si2);                        \
    ((((int64_t)(si2)) == ((int64_t)0)) ||       \
     ((((int64_t)(si1)) == (INT64_MIN)) &&       \
      (((int64_t)(si2)) == ((int64_t)-1))))      \
        ? ((int64_t)(si1))                       \
        : (((int64_t)(si1)) % ((int64_t)(si2))); \
  })

#define safe_div_func_int64_t_s_s(_si1, _si2)    \
  ({                                             \
    int64_t si1 = (_si1);                        \
    int64_t si2 = (_si2);                        \
    ((((int64_t)(si2)) == ((int64_t)0)) ||       \
     ((((int64_t)(si1)) == (INT64_MIN)) &&       \
      (((int64_t)(si2)) == ((int64_t)-1))))      \
        ? ((int64_t)(si1))                       \
        : (((int64_t)(si1)) / ((int64_t)(si2))); \
  })

#define safe_lshift_func_int64_t_s_s(_left, _right)                           \
  ({                                                                          \
    int64_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int64_t)(left)) < ((int64_t)0)) || (((int)(right)) < ((int64_t)0)) || \
     (((int)(right)) >= sizeof(int64_t) * CHAR_BIT) ||                        \
     (((int64_t)(left)) > ((INT64_MAX) >> ((int)(right)))))                   \
        ? ((int64_t)(left))                                                   \
        : (((int64_t)(left)) << ((int)(right)));                              \
  })

#define safe_lshift_func_int64_t_s_u(_left, _right)                  \
  ({                                                                 \
    int64_t left = (_left);                                          \
    unsigned int right = (_right);                                   \
    ((((int64_t)(left)) < ((int64_t)0)) ||                           \
     (((unsigned int)(right)) >= sizeof(int64_t) * CHAR_BIT) ||      \
     (((int64_t)(left)) > ((INT64_MAX) >> ((unsigned int)(right))))) \
        ? ((int64_t)(left))                                          \
        : (((int64_t)(left)) << ((unsigned int)(right)));            \
  })

#define safe_rshift_func_int64_t_s_s(_left, _right)                           \
  ({                                                                          \
    int64_t left = (_left);                                                   \
    int right = (_right);                                                     \
    ((((int64_t)(left)) < ((int64_t)0)) || (((int)(right)) < ((int64_t)0)) || \
     (((int)(right)) >= sizeof(int64_t) * CHAR_BIT))                          \
        ? ((int64_t)(left))                                                   \
        : (((int64_t)(left)) >> ((int)(right)));                              \
  })

#define safe_rshift_func_int64_t_s_u(_left, _right)           \
  ({                                                          \
    int64_t left = (_left);                                   \
    unsigned int right = (_right);                            \
    ((((int64_t)(left)) < ((int64_t)0)) ||                    \
     (((unsigned int)(right)) >= sizeof(int64_t) * CHAR_BIT)) \
        ? ((int64_t)(left))                                   \
        : (((int64_t)(left)) >> ((unsigned int)(right)));     \
  })

#define safe_unary_minus_func_uint8_t_u(_ui) \
  ({                                         \
    uint8_t ui = (_ui);                      \
    -((uint8_t)(ui));                        \
  })

#define safe_add_func_uint8_t_u_u(_ui1, _ui2) \
  ({                                          \
    uint8_t ui1 = (_ui1);                     \
    uint8_t ui2 = (_ui2);                     \
    ((uint8_t)(ui1)) + ((uint8_t)(ui2));      \
  })

#define safe_sub_func_uint8_t_u_u(_ui1, _ui2) \
  ({                                          \
    uint8_t ui1 = (_ui1);                     \
    uint8_t ui2 = (_ui2);                     \
    ((uint8_t)(ui1)) - ((uint8_t)(ui2));      \
  })

#define safe_mul_func_uint8_t_u_u(_ui1, _ui2)                 \
  ({                                                          \
    uint8_t ui1 = (_ui1);                                     \
    uint8_t ui2 = (_ui2);                                     \
    (uint8_t)(((unsigned int)(ui1)) * ((unsigned int)(ui2))); \
  })

#define safe_mod_func_uint8_t_u_u(_ui1, _ui2)    \
  ({                                             \
    uint8_t ui1 = (_ui1);                        \
    uint8_t ui2 = (_ui2);                        \
    (((uint8_t)(ui2)) == ((uint8_t)0))           \
        ? ((uint8_t)(ui1))                       \
        : (((uint8_t)(ui1)) % ((uint8_t)(ui2))); \
  })

#define safe_div_func_uint8_t_u_u(_ui1, _ui2)    \
  ({                                             \
    uint8_t ui1 = (_ui1);                        \
    uint8_t ui2 = (_ui2);                        \
    (((uint8_t)(ui2)) == ((uint8_t)0))           \
        ? ((uint8_t)(ui1))                       \
        : (((uint8_t)(ui1)) / ((uint8_t)(ui2))); \
  })

#define safe_lshift_func_uint8_t_u_s(_left, _right)         \
  ({                                                        \
    uint8_t left = (_left);                                 \
    int right = (_right);                                   \
    ((((int)(right)) < ((uint8_t)0)) ||                     \
     (((int)(right)) >= sizeof(uint8_t) * CHAR_BIT) ||      \
     (((uint8_t)(left)) > ((UINT8_MAX) >> ((int)(right))))) \
        ? ((uint8_t)(left))                                 \
        : (((uint8_t)(left)) << ((int)(right)));            \
  })

#define safe_lshift_func_uint8_t_u_u(_left, _right)                  \
  ({                                                                 \
    uint8_t left = (_left);                                          \
    unsigned int right = (_right);                                   \
    ((((unsigned int)(right)) >= sizeof(uint8_t) * CHAR_BIT) ||      \
     (((uint8_t)(left)) > ((UINT8_MAX) >> ((unsigned int)(right))))) \
        ? ((uint8_t)(left))                                          \
        : (((uint8_t)(left)) << ((unsigned int)(right)));            \
  })

#define safe_rshift_func_uint8_t_u_s(_left, _right)  \
  ({                                                 \
    uint8_t left = (_left);                          \
    int right = (_right);                            \
    ((((int)(right)) < ((uint8_t)0)) ||              \
     (((int)(right)) >= sizeof(uint8_t) * CHAR_BIT)) \
        ? ((uint8_t)(left))                          \
        : (((uint8_t)(left)) >> ((int)(right)));     \
  })

#define safe_rshift_func_uint8_t_u_u(_left, _right)         \
  ({                                                        \
    uint8_t left = (_left);                                 \
    unsigned int right = (_right);                          \
    (((unsigned int)(right)) >= sizeof(uint8_t) * CHAR_BIT) \
        ? ((uint8_t)(left))                                 \
        : (((uint8_t)(left)) >> ((unsigned int)(right)));   \
  })

#define safe_unary_minus_func_uint16_t_u(_ui) \
  ({                                          \
    uint16_t ui = (_ui);                      \
    -((uint16_t)(ui));                        \
  })

#define safe_add_func_uint16_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint16_t ui1 = (_ui1);                     \
    uint16_t ui2 = (_ui2);                     \
    ((uint16_t)(ui1)) + ((uint16_t)(ui2));     \
  })

#define safe_sub_func_uint16_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint16_t ui1 = (_ui1);                     \
    uint16_t ui2 = (_ui2);                     \
    ((uint16_t)(ui1)) - ((uint16_t)(ui2));     \
  })

#define safe_mul_func_uint16_t_u_u(_ui1, _ui2)                 \
  ({                                                           \
    uint16_t ui1 = (_ui1);                                     \
    uint16_t ui2 = (_ui2);                                     \
    (uint16_t)(((unsigned int)(ui1)) * ((unsigned int)(ui2))); \
  })

#define safe_mod_func_uint16_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint16_t ui1 = (_ui1);                         \
    uint16_t ui2 = (_ui2);                         \
    (((uint16_t)(ui2)) == ((uint16_t)0))           \
        ? ((uint16_t)(ui1))                        \
        : (((uint16_t)(ui1)) % ((uint16_t)(ui2))); \
  })

#define safe_div_func_uint16_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint16_t ui1 = (_ui1);                         \
    uint16_t ui2 = (_ui2);                         \
    (((uint16_t)(ui2)) == ((uint16_t)0))           \
        ? ((uint16_t)(ui1))                        \
        : (((uint16_t)(ui1)) / ((uint16_t)(ui2))); \
  })

#define safe_lshift_func_uint16_t_u_s(_left, _right)          \
  ({                                                          \
    uint16_t left = (_left);                                  \
    int right = (_right);                                     \
    ((((int)(right)) < ((uint16_t)0)) ||                      \
     (((int)(right)) >= sizeof(uint16_t) * CHAR_BIT) ||       \
     (((uint16_t)(left)) > ((UINT16_MAX) >> ((int)(right))))) \
        ? ((uint16_t)(left))                                  \
        : (((uint16_t)(left)) << ((int)(right)));             \
  })

#define safe_lshift_func_uint16_t_u_u(_left, _right)                   \
  ({                                                                   \
    uint16_t left = (_left);                                           \
    unsigned int right = (_right);                                     \
    ((((unsigned int)(right)) >= sizeof(uint16_t) * CHAR_BIT) ||       \
     (((uint16_t)(left)) > ((UINT16_MAX) >> ((unsigned int)(right))))) \
        ? ((uint16_t)(left))                                           \
        : (((uint16_t)(left)) << ((unsigned int)(right)));             \
  })

#define safe_rshift_func_uint16_t_u_s(_left, _right)  \
  ({                                                  \
    uint16_t left = (_left);                          \
    int right = (_right);                             \
    ((((int)(right)) < ((uint16_t)0)) ||              \
     (((int)(right)) >= sizeof(uint16_t) * CHAR_BIT)) \
        ? ((uint16_t)(left))                          \
        : (((uint16_t)(left)) >> ((int)(right)));     \
  })

#define safe_rshift_func_uint16_t_u_u(_left, _right)         \
  ({                                                         \
    uint16_t left = (_left);                                 \
    unsigned int right = (_right);                           \
    (((unsigned int)(right)) >= sizeof(uint16_t) * CHAR_BIT) \
        ? ((uint16_t)(left))                                 \
        : (((uint16_t)(left)) >> ((unsigned int)(right)));   \
  })

#define safe_unary_minus_func_uint32_t_u(_ui) \
  ({                                          \
    uint32_t ui = (_ui);                      \
    -((uint32_t)(ui));                        \
  })

#define safe_add_func_uint32_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint32_t ui1 = (_ui1);                     \
    uint32_t ui2 = (_ui2);                     \
    ((uint32_t)(ui1)) + ((uint32_t)(ui2));     \
  })

#define safe_sub_func_uint32_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint32_t ui1 = (_ui1);                     \
    uint32_t ui2 = (_ui2);                     \
    ((uint32_t)(ui1)) - ((uint32_t)(ui2));     \
  })

#define safe_mul_func_uint32_t_u_u(_ui1, _ui2)                 \
  ({                                                           \
    uint32_t ui1 = (_ui1);                                     \
    uint32_t ui2 = (_ui2);                                     \
    (uint32_t)(((unsigned int)(ui1)) * ((unsigned int)(ui2))); \
  })

#define safe_mod_func_uint32_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint32_t ui1 = (_ui1);                         \
    uint32_t ui2 = (_ui2);                         \
    (((uint32_t)(ui2)) == ((uint32_t)0))           \
        ? ((uint32_t)(ui1))                        \
        : (((uint32_t)(ui1)) % ((uint32_t)(ui2))); \
  })

#define safe_div_func_uint32_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint32_t ui1 = (_ui1);                         \
    uint32_t ui2 = (_ui2);                         \
    (((uint32_t)(ui2)) == ((uint32_t)0))           \
        ? ((uint32_t)(ui1))                        \
        : (((uint32_t)(ui1)) / ((uint32_t)(ui2))); \
  })

#define safe_lshift_func_uint32_t_u_s(_left, _right)          \
  ({                                                          \
    uint32_t left = (_left);                                  \
    int right = (_right);                                     \
    ((((int)(right)) < ((uint32_t)0)) ||                      \
     (((int)(right)) >= sizeof(uint32_t) * CHAR_BIT) ||       \
     (((uint32_t)(left)) > ((UINT32_MAX) >> ((int)(right))))) \
        ? ((uint32_t)(left))                                  \
        : (((uint32_t)(left)) << ((int)(right)));             \
  })

#define safe_lshift_func_uint32_t_u_u(_left, _right)                   \
  ({                                                                   \
    uint32_t left = (_left);                                           \
    unsigned int right = (_right);                                     \
    ((((unsigned int)(right)) >= sizeof(uint32_t) * CHAR_BIT) ||       \
     (((uint32_t)(left)) > ((UINT32_MAX) >> ((unsigned int)(right))))) \
        ? ((uint32_t)(left))                                           \
        : (((uint32_t)(left)) << ((unsigned int)(right)));             \
  })

#define safe_rshift_func_uint32_t_u_s(_left, _right)  \
  ({                                                  \
    uint32_t left = (_left);                          \
    int right = (_right);                             \
    ((((int)(right)) < ((uint32_t)0)) ||              \
     (((int)(right)) >= sizeof(uint32_t) * CHAR_BIT)) \
        ? ((uint32_t)(left))                          \
        : (((uint32_t)(left)) >> ((int)(right)));     \
  })

#define safe_rshift_func_uint32_t_u_u(_left, _right)         \
  ({                                                         \
    uint32_t left = (_left);                                 \
    unsigned int right = (_right);                           \
    (((unsigned int)(right)) >= sizeof(uint32_t) * CHAR_BIT) \
        ? ((uint32_t)(left))                                 \
        : (((uint32_t)(left)) >> ((unsigned int)(right)));   \
  })

#define safe_unary_minus_func_uint64_t_u(_ui) \
  ({                                          \
    uint64_t ui = (_ui);                      \
    -((uint64_t)(ui));                        \
  })

#define safe_add_func_uint64_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint64_t ui1 = (_ui1);                     \
    uint64_t ui2 = (_ui2);                     \
    ((uint64_t)(ui1)) + ((uint64_t)(ui2));     \
  })

#define safe_sub_func_uint64_t_u_u(_ui1, _ui2) \
  ({                                           \
    uint64_t ui1 = (_ui1);                     \
    uint64_t ui2 = (_ui2);                     \
    ((uint64_t)(ui1)) - ((uint64_t)(ui2));     \
  })

#define safe_mul_func_uint64_t_u_u(_ui1, _ui2)                   \
  ({                                                             \
    uint64_t ui1 = (_ui1);                                       \
    uint64_t ui2 = (_ui2);                                       \
    (uint64_t)(((unsigned long)(ui1)) * ((unsigned long)(ui2))); \
  })

#define safe_mod_func_uint64_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint64_t ui1 = (_ui1);                         \
    uint64_t ui2 = (_ui2);                         \
    (((uint64_t)(ui2)) == ((uint64_t)0))           \
        ? ((uint64_t)(ui1))                        \
        : (((uint64_t)(ui1)) % ((uint64_t)(ui2))); \
  })

#define safe_div_func_uint64_t_u_u(_ui1, _ui2)     \
  ({                                               \
    uint64_t ui1 = (_ui1);                         \
    uint64_t ui2 = (_ui2);                         \
    (((uint64_t)(ui2)) == ((uint64_t)0))           \
        ? ((uint64_t)(ui1))                        \
        : (((uint64_t)(ui1)) / ((uint64_t)(ui2))); \
  })

#define safe_lshift_func_uint64_t_u_s(_left, _right)          \
  ({                                                          \
    uint64_t left = (_left);                                  \
    int right = (_right);                                     \
    ((((int)(right)) < ((uint64_t)0)) ||                      \
     (((int)(right)) >= sizeof(uint64_t) * CHAR_BIT) ||       \
     (((uint64_t)(left)) > ((UINT64_MAX) >> ((int)(right))))) \
        ? ((uint64_t)(left))                                  \
        : (((uint64_t)(left)) << ((int)(right)));             \
  })

#define safe_lshift_func_uint64_t_u_u(_left, _right)                   \
  ({                                                                   \
    uint64_t left = (_left);                                           \
    unsigned int right = (_right);                                     \
    ((((unsigned int)(right)) >= sizeof(uint64_t) * CHAR_BIT) ||       \
     (((uint64_t)(left)) > ((UINT64_MAX) >> ((unsigned int)(right))))) \
        ? ((uint64_t)(left))                                           \
        : (((uint64_t)(left)) << ((unsigned int)(right)));             \
  })

#define safe_rshift_func_uint64_t_u_s(_left, _right)  \
  ({                                                  \
    uint64_t left = (_left);                          \
    int right = (_right);                             \
    ((((int)(right)) < ((uint64_t)0)) ||              \
     (((int)(right)) >= sizeof(uint64_t) * CHAR_BIT)) \
        ? ((uint64_t)(left))                          \
        : (((uint64_t)(left)) >> ((int)(right)));     \
  })

#define safe_rshift_func_uint64_t_u_u(_left, _right)         \
  ({                                                         \
    uint64_t left = (_left);                                 \
    unsigned int right = (_right);                           \
    (((unsigned int)(right)) >= sizeof(uint64_t) * CHAR_BIT) \
        ? ((uint64_t)(left))                                 \
        : (((uint64_t)(left)) >> ((unsigned int)(right)));   \
  })

#ifdef NO_ATOMICS
#define atomic_inc(x) -1
#define atomic_add(x, y) (1 + 1)
#define atomic_sub(x, y) (1 + 1)
#define atomic_min(x, y) (1 + 1)
#define atomic_max(x, y) (1 + 1)
#define atomic_and(x, y) (1 + 1)
#define atomic_or(x, y) (1 + 1)
#define atomic_xor(x, y) (1 + 1)
#define atomic_noop() /* for sanity checking */
#endif

inline __attribute__((always_inline)) void transparent_crc_no_string(
    uint64_t *crc64_context, uint64_t val) {
  *crc64_context += val;
}

#define transparent_crc_(A, B, C, D) transparent_crc_no_string(A, B)

inline __attribute__((always_inline)) uint32_t get_linear_group_id(void) {
  return (get_group_id(2) * get_num_groups(1) + get_group_id(1)) *
             get_num_groups(0) +
         get_group_id(0);
}

inline __attribute__((always_inline)) uint32_t get_linear_global_id(void) {
  return (get_global_id(2) * get_global_size(1) + get_global_id(1)) *
             get_global_size(0) +
         get_global_id(0);
}

inline __attribute__((always_inline)) uint32_t get_linear_local_id(void) {
  return (get_local_id(2) * get_local_size(1) + get_local_id(1)) *
             get_local_size(0) +
         get_local_id(0);
}
