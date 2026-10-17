# #DF89E case studies (from the RQ4 sample)

Three cases illustrate Finding 4: slices can focus a model on a flaw it overlooked (case 1) or remove the guard that makes code safe (case 2), but neither helps when the deciding evidence lives in another function (case 3).

**Sources:**
- code, slices and model answers: `Docs/df89e_rq4_sample.csv`;
- model identities: the key;
- final labels: `Docs/df89e_rq4_final_labels.csv`;
- CVE metadata and official fixes: `primevul_pairs.jsonl`.

Both coders independently gave all three cases the same primary code. Line numbers are the function's own; the slice shows only the kept lines.

---

## Case 1: The slice focuses the model on the flawed check (CVE-2020-14147, Redis, CWE-190)

**Cases:** C013 (Phi-4) and C097 (Qwen2.5-Coder-14B), on the same unit (PV-10663, vulnerable version).

**Code** (`getnum`, Redis's embedded Lua `struct` library, 13 lines):
```c
1: static int getnum (lua_State *L, const char **fmt, int df) {
2:    if (!isdigit(**fmt))  /* no number? */
3:      return df;  /* return default value */
4:    else {
5:      int a = 0;
6:      do {
7:       if (a > (INT_MAX / 10) || a * 10 > (INT_MAX - (**fmt - '0')))
8:         luaL_error(L, "integral size overflow");
9:        a = a*10 + *((*fmt)++) - '0';
10:      } while (isdigit(**fmt));
11:      return a;
12:   }
13: }
```

**Ground truth.** The labelled lines are 7–8. The official fix (Redis commit 1eb08bc, "Security: update Lua struct package for security") reports "an integer overflow in a bound-checking statement". It removes this check, together with the `lua_State *L` parameter.

**The slice (R4)** keeps lines 1, 2, 5, 7–11, plus SRCVUL metadata (`vrvars: fmt, a, L`; `cfuncs: luaL_error`).

**What happened:**

| Model | Whole function (R1) | Slice (R4) |
|---|---|---|
| Phi-4 | *safe*, no lines | **vulnerable**, cites 7, 9 |
| Qwen2.5-Coder-14B | *safe*, no lines | **vulnerable**, cites **7, 8** (the labelled lines) |

**Why it matters.** Two different models make the same switch. On the whole function, the bound check reads as protection, and both models clear the code. With the early-return path and loop scaffolding removed, both models focus on the arithmetic in the check and flag it. Qwen-14B cites exactly the labelled lines. Both coders coded this as `DM` (missed visible flaw in the whole function) with slice effect `helped`.

**Caveat for the text:** this is one of the 10 sampled "helped" cases. RQ1 shows that, overall, slices do not improve detection, so the case shows *how* a slice can help, not how often.

---

## Case 2: The slice drops the guards and creates a false alarm (CVE-2017-18552, Linux RDS, CWE-787)

**Case:** C061 (Qwen3-Coder-30B), on the **fixed** version (PV-208450, label *safe*).

**Code** (`rds_recv_track_latency`, after the fix, 26 lines; excerpt):
```c
 7:	if (optlen != sizeof(struct rds_rx_trace_so))
 8:		return -EFAULT;
10:	if (copy_from_user(&trace, optval, sizeof(trace)))
11:		return -EFAULT;
13:	if (trace.rx_traces > RDS_MSG_RX_DGRAM_TRACE_MAX)   /* added by the fix */
14:		return -EFAULT;                                  /* added by the fix */
16:	rs->rs_rx_traces = trace.rx_traces;
17:	for (i = 0; i < rs->rs_rx_traces; i++) {
18:		if (trace.rx_trace_pos[i] > RDS_MSG_RX_DGRAM_TRACE_MAX) {
19:			rs->rs_rx_traces = 0;
20:			return -EFAULT;
21:		}
22:		rs->rs_rx_trace[i] = trace.rx_trace_pos[i];
```

**Ground truth.** This is the fixed version. The official fix (commit 780e982) adds the bound check on `trace.rx_traces` (lines 13–14), which caps the loop that writes `rs->rs_rx_trace[i]`.

**The slice (R4)** keeps lines 1, 4, 5, 10, 13, 16–19 and 22. It keeps every *condition* but drops every `return -EFAULT` (lines 8, 11, 14, 20), and with them the length check on `optlen` (lines 7–8).

**What happened:**

| Input | Qwen3-Coder-30B's answer |
|---|---|
| Whole function (R1) | **safe** (correct) |
| Slice (R4) | *vulnerable*, cites 17, 18, 19, 22 (the loop and the array write) |

**Why it matters.** Without the returns, the conditions no longer visibly stop anything. The added bound check on line 13 now appears to fall through into the loop, so the array write on line 22 looks unguarded. The slice removes exactly the statements that made the fixed code safe: control dependence on early returns, not data dependence on the variables, decides safety here. Both coders coded `DG` (guard dropped by the slice), slice effect `hurt`. It is the most frequent cause of false positives on slices (10 of 14).

---

## Case 3: The evidence lives in another function (CVE-2013-6449, OpenSSL, CWE-310)

**Case:** C049 (Phi-4), whole function (R1), vulnerable version (PV-0).

**Code** (`ssl_get_algorithm2`, 8 lines):
```c
1:  long ssl_get_algorithm2(SSL *s)
2:         {
3:         long alg2 = s->s3->tmp.new_cipher->algorithm2;
4:        if (TLS1_get_version(s) >= TLS1_2_VERSION &&
5:             alg2 == (SSL_HANDSHAKE_MAC_DEFAULT|TLS1_PRF))
6:                 return SSL_HANDSHAKE_MAC_SHA256 | TLS1_PRF_SHA256;
7:         return alg2;
8: 	}
```

**Ground truth.** The labelled line is 4. The official fix (commit ca98926) replaces `TLS1_get_version(s) >= TLS1_2_VERSION` with `s->method->version == TLS1_2_VERSION`.

**What happened.** Phi-4 answers *safe* and cites no lines.

**Why it matters.** The flawed comparison is fully visible in eight lines. What decides whether it is wrong is outside the function:
- what `TLS1_get_version(s)` returns, compared with the version of the SSL method in use;
- which protocol versions that method supports.

No intraprocedural representation, whole function or slice, contains that information. Both coders coded `CX` (missing cross-function context). Missing context causes 9 of 13 false negatives on whole functions, and appears in 32 of the 100 layer-1 cases. This is the evidence Aegis-style repository-level traces aim to supply; our R6 probe, which adds only cross-function *names*, does not supply it.

---

## Spare case for the appendix

**C064** (Qwen3-Coder-30B, CVE-2021-20266, rpm, `hdrblobInit`, 35 lines):
- the whole function reads `blob->ei[0]` and `blob->ei[1]` (lines 9–10) before any check of the input size;
- the official fix adds `if (uc && uc < 8)` and a length check;
- the model answers *safe* on the whole function and *vulnerable* on the slice, citing lines 8, 14 and 15. That misses the labelled reads on lines 9–10: the slice fixed detection, but localization remained a near miss.

Both coders coded it `DM` / `helped` with high confidence.
