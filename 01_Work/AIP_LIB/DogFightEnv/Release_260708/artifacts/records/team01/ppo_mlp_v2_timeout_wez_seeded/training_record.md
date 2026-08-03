# Training Record

- Created at: `2026-08-03T15:17:55`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `ppo`

## Observation

- Mode: `tactical16`
- Size: `16`
- Description: Full tactical observation: ownship attitude + speed + altitude + health, relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. All features normalized to [-1, 1]. Observation space bounds: [-1, 1].
- Features:
  - `ownship_roll_norm`
  - `ownship_pitch_norm`
  - `ownship_yaw_norm`
  - `ownship_speed_norm`
  - `ownship_alt_norm`
  - `ownship_health_norm`
  - `delta_n_norm`
  - `delta_e_norm`
  - `delta_d_norm`
  - `ata_norm`
  - `aa_norm`
  - `az_norm`
  - `el_norm`
  - `target_health_norm`
  - `in_wez`
  - `pursuit_score_norm`

## Reward

- Description: Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) + damage differential + low altitude penalty + terminal rewards.
- Step penalty: `-0.01`
- Damage scale: `20.0`
- Pursuit scale: `0.03`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `8000.0`
- Low altitude penalty: `0.2`
- Win reward: `100.0`
- Loss reward: `-100.0`
- Draw reward: `-20.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 100,
  "framework": "torch",
  "num_env_runners": 2,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward",
  "max_engage_time": 120.0,
  "episode_step_limit": 1200,
  "lr": 0.0003,
  "gamma": 0.99,
  "train_batch_size": 1024,
  "minibatch_size": 128,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": null,
  "model_fcnet_hiddens": "256,256",
  "model_fcnet_activation": "relu",
  "model_head_fcnet_hiddens": "",
  "model_head_fcnet_activation": "relu",
  "model_vf_share_layers": "true",
  "network_spec_json": "",
  "use_lstm": false,
  "use_lstm_sac": false,
  "lstm_scope": "actor_only",
  "lstm_cell_size": 64,
  "max_seq_len": 8,
  "debug_io": false,
  "use_lstm_prioritized_replay": false,
  "output_name": "team01",
  "output_tag": "ppo_mlp_v2_timeout_wez_seeded",
  "notes": "PPO v2: weaker broad pursuit reward, staged ATA bonuses, WEZ-center geometry, per-step WEZ hold bonus, overshoot penalty, far-distance penalty, and timeout handling for terminated/truncated episodes.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 50,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 50,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 25,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 25,
  "engagement_log_steps": 1200,
  "engagement_log_episodes": 3,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\ppo_v2_timeout_wez_seeded.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 120.0,
  "episode_step_limit": 1200,
  "step_ratio": 6,
  "reward": {
    "step_penalty": -0.01,
    "low_altitude_threshold_m": 600.0,
    "low_altitude_penalty": 0.2,
    "pursuit_scale": 0.03,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 8000.0,
    "attack_geometry_scale": 0.1,
    "soft_ata_limit_deg": 20.0,
    "soft_aa_limit_deg": 60.0,
    "attack_range_min_m": 152.4,
    "attack_range_best_m": 533.4,
    "attack_range_max_m": 1200.0,
    "ata_stage_max_range_m": 1500.0,
    "ata_under_10_bonus": 0.01,
    "ata_under_5_bonus": 0.03,
    "ata_under_2_bonus": 0.07,
    "ata_under_1_bonus": 0.12,
    "wez_bonus": 0.5,
    "too_close_distance_m": 400.0,
    "too_close_ata_deg": 20.0,
    "too_close_penalty": 0.15,
    "escape_start_m": 3000.0,
    "escape_full_m": 8000.0,
    "escape_penalty_scale": 0.02,
    "damage_scale": 20.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "crash_penalty": -100.0,
    "timeout_penalty": -30.0,
    "draw_reward": -20.0,
    "guard_fail_penalty": -50.0
  },
  "reward_module": "student.my_reward",
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-111.13382752319615`, episode_len_mean=`396.5`
- iter `1`: reward_mean=`-112.82158931714179`, episode_len_mean=`478.0`
- iter `2`: reward_mean=`-112.45971264695777`, episode_len_mean=`450.0`
- iter `3`: reward_mean=`-112.17587315710163`, episode_len_mean=`456.0`
- iter `4`: reward_mean=`-111.33772525602382`, episode_len_mean=`413.25`
- iter `5`: reward_mean=`-111.49507823741087`, episode_len_mean=`394.5`
- iter `6`: reward_mean=`-110.84104000662468`, episode_len_mean=`378.5`
- iter `7`: reward_mean=`-111.782518856356`, episode_len_mean=`406.75`
- iter `8`: reward_mean=`-111.8195283321454`, episode_len_mean=`409.5`
- iter `9`: reward_mean=`-111.24885059940632`, episode_len_mean=`385.5`
- iter `10`: reward_mean=`-113.30530959610547`, episode_len_mean=`455.0`
- iter `11`: reward_mean=`-113.72908791373412`, episode_len_mean=`475.0`
- iter `12`: reward_mean=`-112.50181090514539`, episode_len_mean=`425.5`
- iter `13`: reward_mean=`-114.75173521787707`, episode_len_mean=`471.0`
- iter `14`: reward_mean=`-112.31376994254859`, episode_len_mean=`428.0`
- iter `15`: reward_mean=`-111.52658236368339`, episode_len_mean=`407.75`
- iter `16`: reward_mean=`-111.1186603420592`, episode_len_mean=`394.5`
- iter `17`: reward_mean=`-113.73587328980372`, episode_len_mean=`460.0`
- iter `18`: reward_mean=`-112.57337460755566`, episode_len_mean=`422.0`
- iter `19`: reward_mean=`-113.26329717633772`, episode_len_mean=`452.0`
- iter `20`: reward_mean=`-76.52500468059941`, episode_len_mean=`522.0`
- iter `21`: reward_mean=`-113.7108049802179`, episode_len_mean=`437.5`
- iter `22`: reward_mean=`-115.35136707220992`, episode_len_mean=`483.5`
- iter `23`: reward_mean=`-113.8743222481005`, episode_len_mean=`477.5`
- iter `24`: reward_mean=`-74.82048881026451`, episode_len_mean=`549.0`
- iter `25`: reward_mean=`-112.73300194723201`, episode_len_mean=`439.0`
- iter `26`: reward_mean=`-115.06734752164527`, episode_len_mean=`487.5`
- iter `27`: reward_mean=`-73.46336563389909`, episode_len_mean=`494.5`
- iter `28`: reward_mean=`-113.4226374748168`, episode_len_mean=`453.0`
- iter `29`: reward_mean=`-113.55425041553482`, episode_len_mean=`464.0`
- iter `30`: reward_mean=`-73.26160497286455`, episode_len_mean=`491.5`
- iter `31`: reward_mean=`-117.03239555106681`, episode_len_mean=`544.0`
- iter `32`: reward_mean=`-115.87331759632488`, episode_len_mean=`522.5`
- iter `33`: reward_mean=`-74.17472317895565`, episode_len_mean=`522.0`
- iter `34`: reward_mean=`-114.4931576199058`, episode_len_mean=`479.5`
- iter `35`: reward_mean=`-114.5741718953818`, episode_len_mean=`477.0`
- iter `36`: reward_mean=`-115.8957333756228`, episode_len_mean=`516.75`
- iter `37`: reward_mean=`-34.244942900006805`, episode_len_mean=`568.0`
- iter `38`: reward_mean=`-114.20456203014663`, episode_len_mean=`472.5`
- iter `39`: reward_mean=`-117.15531525743262`, episode_len_mean=`560.0`
- iter `40`: reward_mean=`-35.96660638609931`, episode_len_mean=`568.0`
- iter `41`: reward_mean=`-74.34693299223123`, episode_len_mean=`510.75`
- iter `42`: reward_mean=`-34.26477252910452`, episode_len_mean=`568.0`
- iter `43`: reward_mean=`-119.9292451076278`, episode_len_mean=`516.0`
- iter `44`: reward_mean=`-34.1889777988321`, episode_len_mean=`568.0`
- iter `45`: reward_mean=`-34.23740555622285`, episode_len_mean=`568.0`
- iter `46`: reward_mean=`-74.55449327421941`, episode_len_mean=`535.5`
- iter `47`: reward_mean=`-73.83825431615591`, episode_len_mean=`514.75`
- iter `48`: reward_mean=`-33.51735802775722`, episode_len_mean=`568.0`
- iter `49`: reward_mean=`-74.63323187949433`, episode_len_mean=`505.5`
- iter `50`: reward_mean=`-34.0806732567292`, episode_len_mean=`568.0`
- iter `51`: reward_mean=`-33.9299041906761`, episode_len_mean=`568.0`
- iter `52`: reward_mean=`-33.75693075936407`, episode_len_mean=`568.0`
- iter `53`: reward_mean=`-34.38348838409456`, episode_len_mean=`568.0`
- iter `54`: reward_mean=`-33.16028157493151`, episode_len_mean=`568.0`
- iter `55`: reward_mean=`-34.25925453956117`, episode_len_mean=`568.0`
- iter `56`: reward_mean=`-34.35015420053045`, episode_len_mean=`568.0`
- iter `57`: reward_mean=`-33.7164777909571`, episode_len_mean=`568.0`
- iter `58`: reward_mean=`-34.406417759280046`, episode_len_mean=`568.0`
- iter `59`: reward_mean=`-33.71768588930156`, episode_len_mean=`568.0`
- iter `60`: reward_mean=`-34.39660604505586`, episode_len_mean=`568.0`
- iter `61`: reward_mean=`-34.049377958975`, episode_len_mean=`568.0`
- iter `62`: reward_mean=`-33.83655610200182`, episode_len_mean=`568.0`
- iter `63`: reward_mean=`-34.396617729657464`, episode_len_mean=`568.0`
- iter `64`: reward_mean=`-34.282402492156656`, episode_len_mean=`568.0`
- iter `65`: reward_mean=`-75.3035816895734`, episode_len_mean=`543.5`
- iter `66`: reward_mean=`-33.71195726155923`, episode_len_mean=`568.0`
- iter `67`: reward_mean=`-33.86937144052976`, episode_len_mean=`568.0`
- iter `68`: reward_mean=`-33.9055679966648`, episode_len_mean=`568.0`
- iter `69`: reward_mean=`-43.26345561400396`, episode_len_mean=`568.0`
- iter `70`: reward_mean=`-34.32833203618005`, episode_len_mean=`568.0`
- iter `71`: reward_mean=`-34.35181953264852`, episode_len_mean=`568.0`
- iter `72`: reward_mean=`-34.382209986534214`, episode_len_mean=`568.0`
- iter `73`: reward_mean=`-76.14008725812289`, episode_len_mean=`554.5`
- iter `74`: reward_mean=`-75.70947441779961`, episode_len_mean=`561.0`
- iter `75`: reward_mean=`-34.212023154650495`, episode_len_mean=`568.0`
- iter `76`: reward_mean=`-74.94933189012774`, episode_len_mean=`541.5`
- iter `77`: reward_mean=`-75.9700867083296`, episode_len_mean=`548.5`
- iter `78`: reward_mean=`-114.22264137504438`, episode_len_mean=`445.0`
- iter `79`: reward_mean=`-36.16773241498408`, episode_len_mean=`568.0`
- iter `80`: reward_mean=`-74.97432821701236`, episode_len_mean=`539.0`
- iter `81`: reward_mean=`-34.23994037208227`, episode_len_mean=`568.0`
- iter `82`: reward_mean=`-74.54731484602448`, episode_len_mean=`530.5`
- iter `83`: reward_mean=`-73.3829347761224`, episode_len_mean=`526.0`
- iter `84`: reward_mean=`-34.43875010602016`, episode_len_mean=`568.0`
- iter `85`: reward_mean=`-36.19822015376434`, episode_len_mean=`568.0`
- iter `86`: reward_mean=`-115.11940809569116`, episode_len_mean=`519.0`
- iter `87`: reward_mean=`-34.3572053548679`, episode_len_mean=`568.0`
- iter `88`: reward_mean=`-34.19479565686757`, episode_len_mean=`568.0`
- iter `89`: reward_mean=`-34.21194758531124`, episode_len_mean=`568.0`
- iter `90`: reward_mean=`-76.19457597547853`, episode_len_mean=`567.5`
- iter `91`: reward_mean=`-74.37151046123842`, episode_len_mean=`542.0`
- iter `92`: reward_mean=`-75.18845426405036`, episode_len_mean=`516.5`
- iter `93`: reward_mean=`-41.93396264353987`, episode_len_mean=`568.0`
- iter `94`: reward_mean=`-74.36011426886157`, episode_len_mean=`513.5`
- iter `95`: reward_mean=`-34.33633367321481`, episode_len_mean=`568.0`
- iter `96`: reward_mean=`-34.280366331170356`, episode_len_mean=`568.0`
- iter `97`: reward_mean=`-33.74170805124979`, episode_len_mean=`568.0`
- iter `98`: reward_mean=`-34.35523285028164`, episode_len_mean=`568.0`
- iter `99`: reward_mean=`-33.936269394515705`, episode_len_mean=`568.0`
