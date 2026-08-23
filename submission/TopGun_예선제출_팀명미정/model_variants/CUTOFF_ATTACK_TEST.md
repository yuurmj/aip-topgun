# Cutoff attack variants

All variants use the same SAC policy weights. Only runtime guidance differs.

| Variant | Purpose |
| --- | --- |
| V4_ATTACK_DIRECT | Remove defense and braking; maximize direct pursuit |
| V5_ATTACK_HOLD | Add speed matching only inside the weapon envelope |
| V6_ATTACK_BALANCED | Keep a small emergency defense and late braking |

Run each `AI_DIAG_*.bat` three times against the same cutoff opponent.
Record own HP, target HP, minimum range, minimum ATA while in range, and result.
The first success criterion is target HP below 100, not a full kill.
