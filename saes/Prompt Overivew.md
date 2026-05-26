# High-level structure of the name + birthday prompts

Source: [bio_text.py:26-73](../../Training_On_LM4/data/bio_text.py#L26-L73) (`TEMPLATES_BIRTHDAY`, 46 paraphrases). Bios are rendered by [render_bio](../../Training_On_LM4/data/bio_text.py#L437) from `BioSampler.render` in [util/bio_sampler.py:45](../util/bio_sampler.py#L45).

## Universal scaffolding (true in 100% of templates)
- Every prompt starts with a leading space + `{name}` = `"{first} {middle} {last}"` (so the **subject is always 3 name tokens at the head**).
- Every prompt contains exactly one `{birthday}` slot, which always expands to `"{Month} {Day}, {Year}"`, e.g. `February 18, 1816`. The numeric day and year are always preceded by a month word and a `,`.
- Example: `' Gabriella Ella Rigby was born on February 18, 1816.'`

## Tokens of interest for interp probes

Two natural "anchor" positions for residual-stream / SAE work:

| Anchor | Why it matters |
|---|---|
| `{name}` span (positions 1–3) | Where person identity has to be encoded — the model must commit the lookup key before it sees the predicate. |
| Token immediately **before** `{birthday}` | This is where the model must already have retrieved the date — the very next token is the month. |
| First sub-token of the month (e.g. ` February`) | First place the answer is *produced*. Good for logit-lens / next-token causal tracing. |
| `,` and year tokens | Probe whether year is stored separately from month/day. |

## Distribution of the token directly preceding `{birthday}`

| Preceding word | Count | % of templates |
|---|---:|---:|
| **`on`** | 30 | **65.2%** |
| `of` | 6 | 13.0% |
| `born,` | 2 | 4.3% |
| `day,` | 2 | 4.3% |
| `is`, `recognizes`, `year,`, `as`, `marks`, `acknowledges` | 1 each | 2.2% each |

So **no, not all birthdays appear after "on"** — only ~65%. About 13% follow "of" (`"...birth on the memorable date of {birthday}"`, `"...special day of {birthday} every year"`, etc.), and the remaining ~22% follow a comma, a verb (`recognizes` / `marks` / `acknowledges`), or `is`/`as`.

## Distribution of what follows `{birthday}`

| Following | % |
|---|---:|
| `.` (end of sentence) | 71.7% |
| `,` (more clause follows) | 17.4% |
| nothing (`{birthday}` is final token, no period) | 10.9% |

`{birthday}` is the **last word** of the template in 33/46 (71.7%) cases — a strong positional prior the model can exploit.

## Practical implications for SAE / probe work

- For "where is the date retrieved?" probes, the **single most reliable anchor across templates is the token *before* the month**, because `{birthday}` always immediately precedes the month token. The literal "on"-token alignment only works for ~65% of prompts; if you want one position that's valid on 100%, use `position_of(first month token) − 1`.
- For "where is identity encoded?", the **last name token** (position 3 of the name span) is universally the place where the full identity is available and the predicate hasn't started yet.
- If you stratify activations by *the preceding connector word* (`on` vs `of` vs verb vs comma), you can test whether the model's date-retrieval circuit is anchored to lexical context ("on" specifically) or to a more abstract "next token is a date" signal — the 35% non-`on` templates make this a clean contrast set.
