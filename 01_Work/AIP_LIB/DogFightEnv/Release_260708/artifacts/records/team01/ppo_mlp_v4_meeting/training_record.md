# Training Record

- Created at: `2026-08-03T21:59:14`
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
- Pursuit half angle (deg): `60.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `0.2`
- Win reward: `100.0`
- Loss reward: `-150.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 100,
  "framework": "torch",
  "num_env_runners": 1,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
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
  "output_tag": "ppo_mlp_v4_meeting",
  "notes": "PPO v4 fixed-target attack and WEZ training with v3 logging fields.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 25,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 0,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 25,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 25,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 10,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_v4_meeting.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "reward_module": "student.my_reward",
  "reward": {
    "survival_bonus": 0.0,
    "damage_scale": 20.0,
    "pursuit_half_angle_deg": 60.0,
    "step_penalty": -0.01,
    "pursuit_scale": 0.03,
    "pursuit_angle_deg": 60.0,
    "pursuit_range_m": 3000.0,
    "attack_min_range_m": 500.0,
    "attack_max_range_m": 920.0,
    "attack_ata_deg": 15.0,
    "attack_position_bonus": 0.03,
    "wez_hold_bonus": 0.05,
    "target_damage_scale": 20.0,
    "ownship_damage_scale": 25.0,
    "low_altitude_m": 600.0,
    "low_altitude_penalty": 0.2,
    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.2,
    "win_reward": 100.0,
    "loss_reward": -150.0,
    "crash_penalty": -150.0,
    "guard_fail_penalty": -50.0,
    "timeout_penalty": -30.0,
    "target_crash_reward": 0.0,
    "draw_reward": -10.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-157.39683704525663`, episode_len_mean=`489.0`
- iter `1`: reward_mean=`-156.34313935715258`, episode_len_mean=`375.5`
- iter `2`: reward_mean=`-156.83576320786798`, episode_len_mean=`386.3333333333333`
- iter `3`: reward_mean=`-157.66242063519007`, episode_len_mean=`435.0`
- iter `4`: reward_mean=`-156.44593232715877`, episode_len_mean=`400.0`
- iter `5`: reward_mean=`-156.74401797569084`, episode_len_mean=`433.5`
- iter `6`: reward_mean=`-156.59714934665075`, episode_len_mean=`377.6666666666667`
- iter `7`: reward_mean=`-156.81280325753696`, episode_len_mean=`400.0`
- iter `8`: reward_mean=`-158.38781772372886`, episode_len_mean=`473.3333333333333`
- iter `9`: reward_mean=`-155.74233502175025`, episode_len_mean=`345.3333333333333`
- iter `10`: reward_mean=`-156.4567633691267`, episode_len_mean=`420.5`
- iter `11`: reward_mean=`-158.17269699142915`, episode_len_mean=`496.0`
- iter `12`: reward_mean=`-81.80519342594594`, episode_len_mean=`467.5`
- iter `13`: reward_mean=`-157.29005308309812`, episode_len_mean=`456.5`
- iter `14`: reward_mean=`-106.90440411850868`, episode_len_mean=`464.3333333333333`
- iter `15`: reward_mean=`-82.30115962285142`, episode_len_mean=`488.0`
- iter `16`: reward_mean=`-158.05718154143236`, episode_len_mean=`421.0`
- iter `17`: reward_mean=`-158.64409822259415`, episode_len_mean=`501.0`
- iter `18`: reward_mean=`-82.75408635726791`, episode_len_mean=`529.5`
- iter `19`: reward_mean=`-159.5948809383902`, episode_len_mean=`438.0`
- iter `20`: reward_mean=`-6.786729479656456`, episode_len_mean=`568.0`
- iter `21`: reward_mean=`-83.2783760502486`, episode_len_mean=`565.0`
- iter `22`: reward_mean=`-83.80601529085592`, episode_len_mean=`531.5`
- iter `23`: reward_mean=`-157.21464857016872`, episode_len_mean=`462.0`
- iter `24`: reward_mean=`-6.021504804277002`, episode_len_mean=`568.0`
- iter `25`: reward_mean=`-158.3967153124857`, episode_len_mean=`456.0`
- iter `26`: reward_mean=`-82.11561438636805`, episode_len_mean=`530.0`
- iter `27`: reward_mean=`-82.68135084689497`, episode_len_mean=`524.0`
- iter `28`: reward_mean=`-158.49092451730866`, episode_len_mean=`491.0`
- iter `29`: reward_mean=`-159.111940955806`, episode_len_mean=`490.5`
- iter `30`: reward_mean=`-83.20326477074087`, episode_len_mean=`513.0`
- iter `31`: reward_mean=`-159.59301660628654`, episode_len_mean=`522.5`
- iter `32`: reward_mean=`-82.15298244796874`, episode_len_mean=`555.5`
- iter `33`: reward_mean=`-6.591440084081292`, episode_len_mean=`568.0`
- iter `34`: reward_mean=`-6.82289947692925`, episode_len_mean=`568.0`
- iter `35`: reward_mean=`-6.455521142351959`, episode_len_mean=`568.0`
- iter `36`: reward_mean=`-5.985049587760978`, episode_len_mean=`568.0`
- iter `37`: reward_mean=`-5.85376320011837`, episode_len_mean=`568.0`
- iter `38`: reward_mean=`-6.114305019914415`, episode_len_mean=`568.0`
- iter `39`: reward_mean=`-6.6951175122799444`, episode_len_mean=`568.0`
- iter `40`: reward_mean=`-6.830377746021688`, episode_len_mean=`568.0`
- iter `41`: reward_mean=`-6.562158043623741`, episode_len_mean=`568.0`
- iter `42`: reward_mean=`-6.244728197761933`, episode_len_mean=`568.0`
- iter `43`: reward_mean=`-6.723990302184695`, episode_len_mean=`568.0`
- iter `44`: reward_mean=`-6.426848858046954`, episode_len_mean=`568.0`
- iter `45`: reward_mean=`-5.877351097124857`, episode_len_mean=`568.0`
- iter `46`: reward_mean=`-6.699460676291746`, episode_len_mean=`568.0`
- iter `47`: reward_mean=`-6.082689049393148`, episode_len_mean=`568.0`
- iter `48`: reward_mean=`-6.419577435814633`, episode_len_mean=`568.0`
- iter `49`: reward_mean=`-5.304712823050776`, episode_len_mean=`568.0`
- iter `50`: reward_mean=`-6.752786685019516`, episode_len_mean=`568.0`
- iter `51`: reward_mean=`-6.806733655082023`, episode_len_mean=`568.0`
- iter `52`: reward_mean=`-6.1111668557434795`, episode_len_mean=`568.0`
- iter `53`: reward_mean=`-6.652641334337522`, episode_len_mean=`568.0`
- iter `54`: reward_mean=`-6.579694156637766`, episode_len_mean=`568.0`
- iter `55`: reward_mean=`-5.291560628185383`, episode_len_mean=`568.0`
- iter `56`: reward_mean=`-6.542611430738009`, episode_len_mean=`568.0`
- iter `57`: reward_mean=`-5.499107220537187`, episode_len_mean=`568.0`
- iter `58`: reward_mean=`-6.556008239904761`, episode_len_mean=`568.0`
- iter `59`: reward_mean=`-6.0733044110431305`, episode_len_mean=`568.0`
- iter `60`: reward_mean=`-5.465855896206558`, episode_len_mean=`568.0`
- iter `61`: reward_mean=`-5.6569825111102325`, episode_len_mean=`568.0`
- iter `62`: reward_mean=`-6.023486355095894`, episode_len_mean=`568.0`
- iter `63`: reward_mean=`-6.634019639848019`, episode_len_mean=`568.0`
- iter `64`: reward_mean=`-6.1704400879747965`, episode_len_mean=`568.0`
- iter `65`: reward_mean=`-6.524725591288474`, episode_len_mean=`568.0`
- iter `66`: reward_mean=`-6.0474222914943425`, episode_len_mean=`568.0`
- iter `67`: reward_mean=`-5.999175610942816`, episode_len_mean=`568.0`
- iter `68`: reward_mean=`-5.969593599096544`, episode_len_mean=`568.0`
- iter `69`: reward_mean=`-6.102569647407765`, episode_len_mean=`568.0`
- iter `70`: reward_mean=`-6.432063740466882`, episode_len_mean=`568.0`
- iter `71`: reward_mean=`-5.970016623142267`, episode_len_mean=`568.0`
- iter `72`: reward_mean=`-5.39536247495838`, episode_len_mean=`568.0`
- iter `73`: reward_mean=`-6.203937642311898`, episode_len_mean=`568.0`
- iter `74`: reward_mean=`-5.57098742875505`, episode_len_mean=`568.0`
- iter `75`: reward_mean=`-5.609079723159574`, episode_len_mean=`568.0`
- iter `76`: reward_mean=`-5.840784916404525`, episode_len_mean=`568.0`
- iter `77`: reward_mean=`-6.466109480243664`, episode_len_mean=`568.0`
- iter `78`: reward_mean=`-5.538385263398549`, episode_len_mean=`568.0`
- iter `79`: reward_mean=`-6.082854936808238`, episode_len_mean=`568.0`
- iter `80`: reward_mean=`-5.585573924099469`, episode_len_mean=`568.0`
- iter `81`: reward_mean=`-5.760333956372069`, episode_len_mean=`568.0`
- iter `82`: reward_mean=`-6.1386723196021595`, episode_len_mean=`568.0`
- iter `83`: reward_mean=`-5.441635861746367`, episode_len_mean=`568.0`
- iter `84`: reward_mean=`-5.6117574556078935`, episode_len_mean=`568.0`
- iter `85`: reward_mean=`-6.209884062953421`, episode_len_mean=`568.0`
- iter `86`: reward_mean=`-5.649416236130762`, episode_len_mean=`568.0`
- iter `87`: reward_mean=`-6.21230576249852`, episode_len_mean=`568.0`
- iter `88`: reward_mean=`-5.306983715220575`, episode_len_mean=`568.0`
- iter `89`: reward_mean=`-5.55939527117786`, episode_len_mean=`568.0`
- iter `90`: reward_mean=`-5.651734543494738`, episode_len_mean=`568.0`
- iter `91`: reward_mean=`-5.533344262912534`, episode_len_mean=`568.0`
- iter `92`: reward_mean=`-6.2193864605928475`, episode_len_mean=`568.0`
- iter `93`: reward_mean=`-5.476147370150857`, episode_len_mean=`568.0`
- iter `94`: reward_mean=`-5.494697917983158`, episode_len_mean=`568.0`
- iter `95`: reward_mean=`-5.225234918805469`, episode_len_mean=`568.0`
- iter `96`: reward_mean=`-5.614502839584118`, episode_len_mean=`568.0`
- iter `97`: reward_mean=`-6.103430883653807`, episode_len_mean=`568.0`
- iter `98`: reward_mean=`-5.61792777232872`, episode_len_mean=`568.0`
- iter `99`: reward_mean=`-5.534047616124209`, episode_len_mean=`568.0`
