# Training Record

- Created at: `2026-07-23T19:44:21`
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
- Pursuit scale: `0.06`
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
  "iterations": 300,
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
  "max_engage_time": 120.0,
  "episode_step_limit": 1200,
  "lr": 0.0003,
  "gamma": 0.99,
  "train_batch_size": 4096,
  "minibatch_size": 256,
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
  "output_tag": "ppo_v1_attack_geometry",
  "notes": "PPO MLP v1. Fixed-target attack-geometry shaping with explicit timeout penalty, broad pursuit signal, rear-aspect reward, soft WEZ reward, and close-pass penalty.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 25,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 25,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 10,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 10,
  "engagement_log_steps": 1200,
  "engagement_log_episodes": 3,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\03_Experiments\\baseline\\ppo_v1_attack_geometry.yaml"
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
  "reward_module": "student.my_reward",
  "reward": {
    "step_penalty": -0.01,
    "low_altitude_penalty": 0.2,
    "heading_scale": 0.03,
    "pursuit_scale": 0.06,
    "pursuit_range_m": 8000.0,
    "tail_scale": 0.04,
    "attack_geometry_scale": 0.12,
    "soft_ata_limit_deg": 45.0,
    "soft_aa_limit_deg": 90.0,
    "attack_range_min_m": 300.0,
    "attack_range_best_m": 700.0,
    "attack_range_max_m": 1500.0,
    "wez_bonus": 0.3,
    "too_close_distance_m": 250.0,
    "too_close_ata_deg": 45.0,
    "too_close_penalty": 0.25,
    "damage_scale": 20.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "timeout_penalty": -30.0,
    "draw_reward": -20.0,
    "guard_fail_penalty": -50.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-98.6418147314839`, episode_len_mean=`443.0`
- iter `1`: reward_mean=`-89.57514852188125`, episode_len_mean=`424.3333333333333`
- iter `2`: reward_mean=`-98.38293265338118`, episode_len_mean=`406.5`
- iter `3`: reward_mean=`-90.31472488596289`, episode_len_mean=`423.0`
- iter `4`: reward_mean=`-98.73101012059082`, episode_len_mean=`463.55555555555554`
- iter `5`: reward_mean=`-98.97654363771997`, episode_len_mean=`458.77777777777777`
- iter `6`: reward_mean=`-99.41525218728263`, episode_len_mean=`459.3333333333333`
- iter `7`: reward_mean=`-90.7650275813213`, episode_len_mean=`468.55555555555554`
- iter `8`: reward_mean=`-89.48147078170481`, episode_len_mean=`469.125`
- iter `9`: reward_mean=`-90.85551918520555`, episode_len_mean=`473.6666666666667`
- iter `10`: reward_mean=`-89.66203605284736`, episode_len_mean=`486.625`
- iter `11`: reward_mean=`-100.16208469247437`, episode_len_mean=`455.8888888888889`
- iter `12`: reward_mean=`-91.46605166365276`, episode_len_mean=`493.22222222222223`
- iter `13`: reward_mean=`-79.96823412882244`, episode_len_mean=`501.875`
- iter `14`: reward_mean=`-90.61895901083847`, episode_len_mean=`500.25`
- iter `15`: reward_mean=`-70.00497851308207`, episode_len_mean=`523.625`
- iter `16`: reward_mean=`-59.61265263189509`, episode_len_mean=`527.25`
- iter `17`: reward_mean=`-90.78013769446676`, episode_len_mean=`507.0`
- iter `18`: reward_mean=`-53.52954266605845`, episode_len_mean=`545.4285714285714`
- iter `19`: reward_mean=`-59.35796468380398`, episode_len_mean=`531.125`
- iter `20`: reward_mean=`-29.471107554496484`, episode_len_mean=`561.5714285714286`
- iter `21`: reward_mean=`-42.32315284928325`, episode_len_mean=`558.7142857142857`
- iter `22`: reward_mean=`-38.9671680375766`, episode_len_mean=`556.5`
- iter `23`: reward_mean=`-17.939339463434177`, episode_len_mean=`568.0`
- iter `24`: reward_mean=`-30.55304109308532`, episode_len_mean=`564.0`
- iter `25`: reward_mean=`-29.748620357541338`, episode_len_mean=`567.7142857142857`
- iter `26`: reward_mean=`-20.442495975295877`, episode_len_mean=`568.0`
- iter `27`: reward_mean=`-30.066130734642133`, episode_len_mean=`564.8571428571429`
- iter `28`: reward_mean=`-19.342441898812805`, episode_len_mean=`568.0`
- iter `29`: reward_mean=`-17.70043461619743`, episode_len_mean=`568.0`
- iter `30`: reward_mean=`-28.209638722219`, episode_len_mean=`561.375`
- iter `31`: reward_mean=`-17.541030919624596`, episode_len_mean=`568.0`
- iter `32`: reward_mean=`-17.669762318839155`, episode_len_mean=`568.0`
- iter `33`: reward_mean=`-18.20606195676324`, episode_len_mean=`568.0`
- iter `34`: reward_mean=`-17.87733165039326`, episode_len_mean=`568.0`
- iter `35`: reward_mean=`-18.152004821608475`, episode_len_mean=`568.0`
- iter `36`: reward_mean=`-17.552885985892903`, episode_len_mean=`568.0`
- iter `37`: reward_mean=`-18.55054208064151`, episode_len_mean=`568.0`
- iter `38`: reward_mean=`-17.700180893044607`, episode_len_mean=`568.0`
- iter `39`: reward_mean=`-17.657201463804128`, episode_len_mean=`568.0`
- iter `40`: reward_mean=`-18.51992911455042`, episode_len_mean=`568.0`
- iter `41`: reward_mean=`-17.575664308683816`, episode_len_mean=`568.0`
- iter `42`: reward_mean=`-17.445367020812565`, episode_len_mean=`568.0`
- iter `43`: reward_mean=`-17.67216208477959`, episode_len_mean=`568.0`
- iter `44`: reward_mean=`-17.55867181095399`, episode_len_mean=`568.0`
- iter `45`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `46`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `47`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `48`: reward_mean=`-17.726521212882155`, episode_len_mean=`568.0`
- iter `49`: reward_mean=`-17.40855782571559`, episode_len_mean=`568.0`
- iter `50`: reward_mean=`-17.47216491781556`, episode_len_mean=`568.0`
- iter `51`: reward_mean=`-17.36888525968834`, episode_len_mean=`568.0`
- iter `52`: reward_mean=`-17.426335478927374`, episode_len_mean=`568.0`
- iter `53`: reward_mean=`-17.118731540870037`, episode_len_mean=`568.0`
- iter `54`: reward_mean=`-17.194011276323472`, episode_len_mean=`568.0`
- iter `55`: reward_mean=`-28.81683479668545`, episode_len_mean=`566.2857142857143`
- iter `56`: reward_mean=`-17.197840112680634`, episode_len_mean=`568.0`
- iter `57`: reward_mean=`-18.406075409483584`, episode_len_mean=`568.0`
- iter `58`: reward_mean=`-16.760424163221124`, episode_len_mean=`568.0`
- iter `59`: reward_mean=`-17.132916962526448`, episode_len_mean=`568.0`
- iter `60`: reward_mean=`-16.645797699276716`, episode_len_mean=`568.0`
- iter `61`: reward_mean=`-17.136321799164005`, episode_len_mean=`568.0`
- iter `62`: reward_mean=`-19.727879391039917`, episode_len_mean=`568.0`
- iter `63`: reward_mean=`-17.521089288505046`, episode_len_mean=`568.0`
- iter `64`: reward_mean=`-17.35471919175702`, episode_len_mean=`568.0`
- iter `65`: reward_mean=`-16.840732582972294`, episode_len_mean=`568.0`
- iter `66`: reward_mean=`-17.582551810973218`, episode_len_mean=`568.0`
- iter `67`: reward_mean=`-27.688975569919023`, episode_len_mean=`566.125`
- iter `68`: reward_mean=`-17.10179259690133`, episode_len_mean=`568.0`
- iter `69`: reward_mean=`-18.72684148621488`, episode_len_mean=`568.0`
- iter `70`: reward_mean=`-17.3412433749612`, episode_len_mean=`568.0`
- iter `71`: reward_mean=`-16.990503880151266`, episode_len_mean=`568.0`
- iter `72`: reward_mean=`-49.106781955098825`, episode_len_mean=`564.75`
- iter `73`: reward_mean=`-52.54511914016585`, episode_len_mean=`563.8571428571429`
- iter `74`: reward_mean=`-29.901182506090077`, episode_len_mean=`564.2857142857143`
- iter `75`: reward_mean=`-59.55735524848906`, episode_len_mean=`552.875`
- iter `76`: reward_mean=`-31.637004476811164`, episode_len_mean=`563.4285714285714`
- iter `77`: reward_mean=`-28.323691609363312`, episode_len_mean=`558.8571428571429`
- iter `78`: reward_mean=`-29.783277503028142`, episode_len_mean=`561.1428571428571`
- iter `79`: reward_mean=`-16.73931998291596`, episode_len_mean=`568.0`
- iter `80`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `81`: reward_mean=`-16.931932830137445`, episode_len_mean=`568.0`
- iter `82`: reward_mean=`-28.572914154789803`, episode_len_mean=`561.8571428571429`
- iter `83`: reward_mean=`-16.636562837825934`, episode_len_mean=`568.0`
- iter `84`: reward_mean=`-16.62935421931212`, episode_len_mean=`568.0`
- iter `85`: reward_mean=`-17.20800265445817`, episode_len_mean=`568.0`
- iter `86`: reward_mean=`-29.18170547549305`, episode_len_mean=`566.0`
- iter `87`: reward_mean=`-37.09098806312723`, episode_len_mean=`559.25`
- iter `88`: reward_mean=`-17.083556790004916`, episode_len_mean=`568.0`
- iter `89`: reward_mean=`-16.63960930737891`, episode_len_mean=`568.0`
- iter `90`: reward_mean=`-16.21582710681286`, episode_len_mean=`568.0`
- iter `91`: reward_mean=`-16.67856418918716`, episode_len_mean=`568.0`
- iter `92`: reward_mean=`-26.396092863252296`, episode_len_mean=`560.75`
- iter `93`: reward_mean=`-16.07194104755157`, episode_len_mean=`568.0`
- iter `94`: reward_mean=`-15.45174762564994`, episode_len_mean=`568.0`
- iter `95`: reward_mean=`-16.324259617696214`, episode_len_mean=`568.0`
- iter `96`: reward_mean=`-16.11747138904947`, episode_len_mean=`568.0`
- iter `97`: reward_mean=`-16.832631455012237`, episode_len_mean=`568.0`
- iter `98`: reward_mean=`-15.743572438130258`, episode_len_mean=`568.0`
- iter `99`: reward_mean=`-18.406524623509465`, episode_len_mean=`568.0`
- iter `100`: reward_mean=`-15.630758268700607`, episode_len_mean=`568.0`
- iter `101`: reward_mean=`-17.298324222619293`, episode_len_mean=`568.0`
- iter `102`: reward_mean=`-16.449716483240955`, episode_len_mean=`568.0`
- iter `103`: reward_mean=`-15.842610863692324`, episode_len_mean=`568.0`
- iter `104`: reward_mean=`-15.296773187558328`, episode_len_mean=`568.0`
- iter `105`: reward_mean=`-26.326266298615174`, episode_len_mean=`567.375`
- iter `106`: reward_mean=`-14.091395767379792`, episode_len_mean=`568.0`
- iter `107`: reward_mean=`-14.906076017941317`, episode_len_mean=`568.0`
- iter `108`: reward_mean=`-16.828402832450834`, episode_len_mean=`568.0`
- iter `109`: reward_mean=`-15.179807327151455`, episode_len_mean=`568.0`
- iter `110`: reward_mean=`-25.374802792509954`, episode_len_mean=`562.75`
- iter `111`: reward_mean=`-14.644543703677263`, episode_len_mean=`568.0`
- iter `112`: reward_mean=`-27.273853893013523`, episode_len_mean=`562.2857142857143`
- iter `113`: reward_mean=`-13.781765806463834`, episode_len_mean=`568.0`
- iter `114`: reward_mean=`-36.75259155895368`, episode_len_mean=`555.0`
- iter `115`: reward_mean=`-14.014805031482798`, episode_len_mean=`568.0`
- iter `116`: reward_mean=`-15.187178706650911`, episode_len_mean=`568.0`
- iter `117`: reward_mean=`-15.24652236679274`, episode_len_mean=`568.0`
- iter `118`: reward_mean=`-14.860222054260838`, episode_len_mean=`568.0`
- iter `119`: reward_mean=`-13.264144889804088`, episode_len_mean=`568.0`
- iter `120`: reward_mean=`-51.08770948454968`, episode_len_mean=`539.0`
- iter `121`: reward_mean=`-36.409988222821084`, episode_len_mean=`562.25`
- iter `122`: reward_mean=`-13.238778061536541`, episode_len_mean=`568.0`
- iter `123`: reward_mean=`-25.937147284394456`, episode_len_mean=`564.2857142857143`
- iter `124`: reward_mean=`-15.33585298101392`, episode_len_mean=`568.0`
- iter `125`: reward_mean=`-24.606946899501576`, episode_len_mean=`558.625`
- iter `126`: reward_mean=`-13.331054320995452`, episode_len_mean=`568.0`
- iter `127`: reward_mean=`-12.381171552411008`, episode_len_mean=`568.0`
- iter `128`: reward_mean=`-23.523757479502756`, episode_len_mean=`545.5`
- iter `129`: reward_mean=`-12.307885514813677`, episode_len_mean=`568.0`
- iter `130`: reward_mean=`-24.63829168882163`, episode_len_mean=`545.8571428571429`
- iter `131`: reward_mean=`-12.03172081831813`, episode_len_mean=`568.0`
- iter `132`: reward_mean=`-10.80162796292413`, episode_len_mean=`568.0`
- iter `133`: reward_mean=`-12.650920453558765`, episode_len_mean=`568.0`
- iter `134`: reward_mean=`-22.54165485555179`, episode_len_mean=`558.1428571428571`
- iter `135`: reward_mean=`-11.310276527069078`, episode_len_mean=`568.0`
- iter `136`: reward_mean=`-23.283308708294864`, episode_len_mean=`548.2857142857143`
- iter `137`: reward_mean=`-9.011979102975474`, episode_len_mean=`568.0`
- iter `138`: reward_mean=`-7.863767138117311`, episode_len_mean=`568.0`
- iter `139`: reward_mean=`-22.14346961277799`, episode_len_mean=`557.5`
- iter `140`: reward_mean=`-8.162680662556733`, episode_len_mean=`568.0`
- iter `141`: reward_mean=`-10.584787599564667`, episode_len_mean=`568.0`
- iter `142`: reward_mean=`-9.018119654475978`, episode_len_mean=`568.0`
- iter `143`: reward_mean=`-6.1309231424510955`, episode_len_mean=`568.0`
- iter `144`: reward_mean=`-5.8797583502909845`, episode_len_mean=`568.0`
- iter `145`: reward_mean=`-7.907621302029068`, episode_len_mean=`568.0`
- iter `146`: reward_mean=`-3.7164275296681724`, episode_len_mean=`568.0`
- iter `147`: reward_mean=`-4.883229218951408`, episode_len_mean=`568.0`
- iter `148`: reward_mean=`-2.7037966575168335`, episode_len_mean=`568.0`
- iter `149`: reward_mean=`-2.3126857122227498`, episode_len_mean=`568.0`
- iter `150`: reward_mean=`-16.309715875438332`, episode_len_mean=`565.4285714285714`
- iter `151`: reward_mean=`-1.435322314629156`, episode_len_mean=`568.0`
- iter `152`: reward_mean=`-15.231716765527619`, episode_len_mean=`549.125`
- iter `153`: reward_mean=`-2.0142527557015066`, episode_len_mean=`568.0`
- iter `154`: reward_mean=`2.4553193871849004`, episode_len_mean=`568.0`
- iter `155`: reward_mean=`-13.042086822149585`, episode_len_mean=`546.5`
- iter `156`: reward_mean=`1.3905849382292719`, episode_len_mean=`568.0`
- iter `157`: reward_mean=`0.6545897834448553`, episode_len_mean=`568.0`
- iter `158`: reward_mean=`-0.8104253730279554`, episode_len_mean=`568.0`
- iter `159`: reward_mean=`-2.514591400813221`, episode_len_mean=`568.0`
- iter `160`: reward_mean=`-2.2758614557304724`, episode_len_mean=`568.0`
- iter `161`: reward_mean=`-10.305328278140982`, episode_len_mean=`551.2857142857143`
- iter `162`: reward_mean=`3.7040227692713126`, episode_len_mean=`568.0`
- iter `163`: reward_mean=`-8.880119953689467`, episode_len_mean=`553.375`
- iter `164`: reward_mean=`-0.14945474750085985`, episode_len_mean=`568.0`
- iter `165`: reward_mean=`1.8818310356942776`, episode_len_mean=`568.0`
- iter `166`: reward_mean=`-10.578943588146316`, episode_len_mean=`549.0`
- iter `167`: reward_mean=`3.2580529753198233`, episode_len_mean=`568.0`
- iter `168`: reward_mean=`1.1024269489997618`, episode_len_mean=`568.0`
- iter `169`: reward_mean=`6.496614253934244`, episode_len_mean=`568.0`
- iter `170`: reward_mean=`-10.892106604408564`, episode_len_mean=`542.75`
- iter `171`: reward_mean=`5.528402351716601`, episode_len_mean=`568.0`
- iter `172`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `173`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `174`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `175`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `176`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `177`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `178`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `179`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `180`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `181`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `182`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `183`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `184`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `185`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `186`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `187`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `188`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `189`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `190`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `191`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `192`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `193`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `194`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `195`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `196`: reward_mean=`5.264375528789955`, episode_len_mean=`568.0`
- iter `197`: reward_mean=`5.814707201085708`, episode_len_mean=`568.0`
- iter `198`: reward_mean=`7.718925249899355`, episode_len_mean=`568.0`
- iter `199`: reward_mean=`7.536343787930227`, episode_len_mean=`568.0`
- iter `200`: reward_mean=`7.6105656055806845`, episode_len_mean=`568.0`
- iter `201`: reward_mean=`8.349975713215638`, episode_len_mean=`568.0`
- iter `202`: reward_mean=`8.08198815030866`, episode_len_mean=`568.0`
- iter `203`: reward_mean=`7.232004740841627`, episode_len_mean=`568.0`
- iter `204`: reward_mean=`7.874542465519438`, episode_len_mean=`568.0`
- iter `205`: reward_mean=`7.22959140362772`, episode_len_mean=`568.0`
- iter `206`: reward_mean=`9.405189504007742`, episode_len_mean=`568.0`
- iter `207`: reward_mean=`9.291560471949577`, episode_len_mean=`568.0`
- iter `208`: reward_mean=`5.655077236351517`, episode_len_mean=`568.0`
- iter `209`: reward_mean=`8.814336871674907`, episode_len_mean=`568.0`
- iter `210`: reward_mean=`6.955721317784801`, episode_len_mean=`568.0`
- iter `211`: reward_mean=`6.130199560076347`, episode_len_mean=`568.0`
- iter `212`: reward_mean=`8.691708723851459`, episode_len_mean=`568.0`
- iter `213`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `214`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `215`: reward_mean=`7.312508483767433`, episode_len_mean=`568.0`
- iter `216`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `217`: reward_mean=`8.8820168264651`, episode_len_mean=`568.0`
- iter `218`: reward_mean=`9.818409810429921`, episode_len_mean=`568.0`
- iter `219`: reward_mean=`7.715586837569518`, episode_len_mean=`568.0`
- iter `220`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `221`: reward_mean=`9.31727814948678`, episode_len_mean=`568.0`
- iter `222`: reward_mean=`8.87704519352669`, episode_len_mean=`568.0`
- iter `223`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `224`: reward_mean=`9.885021622840071`, episode_len_mean=`568.0`
- iter `225`: reward_mean=`7.422560774664595`, episode_len_mean=`568.0`
- iter `226`: reward_mean=`10.432060972396254`, episode_len_mean=`568.0`
- iter `227`: reward_mean=`9.911391769020442`, episode_len_mean=`568.0`
- iter `228`: reward_mean=`9.627787215242758`, episode_len_mean=`568.0`
- iter `229`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `230`: reward_mean=`10.485196622977679`, episode_len_mean=`568.0`
- iter `231`: reward_mean=`9.720358931492672`, episode_len_mean=`568.0`
- iter `232`: reward_mean=`-5.065300007233246`, episode_len_mean=`538.5714285714286`
- iter `233`: reward_mean=`10.50580660634856`, episode_len_mean=`568.0`
- iter `234`: reward_mean=`10.516277621826934`, episode_len_mean=`568.0`
- iter `235`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `236`: reward_mean=`10.102625666619574`, episode_len_mean=`568.0`
- iter `237`: reward_mean=`9.97965020269488`, episode_len_mean=`568.0`
- iter `238`: reward_mean=`7.284147807536231`, episode_len_mean=`568.0`
- iter `239`: reward_mean=`9.433488390805483`, episode_len_mean=`568.0`
- iter `240`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `241`: reward_mean=`9.227626084403889`, episode_len_mean=`568.0`
- iter `242`: reward_mean=`7.800658738439755`, episode_len_mean=`568.0`
- iter `243`: reward_mean=`10.133583223611911`, episode_len_mean=`568.0`
- iter `244`: reward_mean=`8.631801689756275`, episode_len_mean=`568.0`
- iter `245`: reward_mean=`7.36267290235936`, episode_len_mean=`568.0`
- iter `246`: reward_mean=`9.456147083512954`, episode_len_mean=`568.0`
- iter `247`: reward_mean=`-3.9245022872441817`, episode_len_mean=`545.75`
- iter `248`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `249`: reward_mean=`10.309451040791346`, episode_len_mean=`568.0`
- iter `250`: reward_mean=`10.550529458969265`, episode_len_mean=`568.0`
- iter `251`: reward_mean=`10.72258264616467`, episode_len_mean=`568.0`
- iter `252`: reward_mean=`9.748329943586066`, episode_len_mean=`568.0`
- iter `253`: reward_mean=`8.83256992742062`, episode_len_mean=`568.0`
- iter `254`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `255`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `256`: reward_mean=`10.083558762495372`, episode_len_mean=`568.0`
- iter `257`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `258`: reward_mean=`-6.319607746113102`, episode_len_mean=`547.8571428571429`
- iter `259`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `260`: reward_mean=`9.99300239030425`, episode_len_mean=`568.0`
- iter `261`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `262`: reward_mean=`-4.977075043249352`, episode_len_mean=`544.875`
- iter `263`: reward_mean=`9.616623017683944`, episode_len_mean=`568.0`
- iter `264`: reward_mean=`9.831972452665777`, episode_len_mean=`568.0`
- iter `265`: reward_mean=`9.31943636824675`, episode_len_mean=`568.0`
- iter `266`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `267`: reward_mean=`9.06745183204809`, episode_len_mean=`568.0`
- iter `268`: reward_mean=`10.901989052872313`, episode_len_mean=`568.0`
- iter `269`: reward_mean=`10.968559226370214`, episode_len_mean=`568.0`
- iter `270`: reward_mean=`10.702950316931322`, episode_len_mean=`568.0`
- iter `271`: reward_mean=`10.136389935087465`, episode_len_mean=`568.0`
- iter `272`: reward_mean=`11.602951610306139`, episode_len_mean=`568.0`
- iter `273`: reward_mean=`10.559285119943997`, episode_len_mean=`568.0`
- iter `274`: reward_mean=`11.52502960554809`, episode_len_mean=`568.0`
- iter `275`: reward_mean=`12.299544525929273`, episode_len_mean=`568.0`
- iter `276`: reward_mean=`11.276612719064277`, episode_len_mean=`568.0`
- iter `277`: reward_mean=`12.340137753097261`, episode_len_mean=`568.0`
- iter `278`: reward_mean=`11.329227794780975`, episode_len_mean=`568.0`
- iter `279`: reward_mean=`8.393275754645407`, episode_len_mean=`568.0`
- iter `280`: reward_mean=`10.462955380096682`, episode_len_mean=`568.0`
- iter `281`: reward_mean=`12.275659364443511`, episode_len_mean=`568.0`
- iter `282`: reward_mean=`11.215418915651705`, episode_len_mean=`568.0`
- iter `283`: reward_mean=`12.978985658070135`, episode_len_mean=`568.0`
- iter `284`: reward_mean=`11.722129078580553`, episode_len_mean=`568.0`
- iter `285`: reward_mean=`11.260094773870964`, episode_len_mean=`568.0`
- iter `286`: reward_mean=`11.618174194435865`, episode_len_mean=`568.0`
- iter `287`: reward_mean=`10.75299238348089`, episode_len_mean=`568.0`
- iter `288`: reward_mean=`11.071544954714884`, episode_len_mean=`568.0`
- iter `289`: reward_mean=`11.321780999303684`, episode_len_mean=`568.0`
- iter `290`: reward_mean=`10.88997339814619`, episode_len_mean=`568.0`
- iter `291`: reward_mean=`12.466207938622704`, episode_len_mean=`568.0`
- iter `292`: reward_mean=`12.333841658020622`, episode_len_mean=`568.0`
- iter `293`: reward_mean=`11.669086589918312`, episode_len_mean=`568.0`
- iter `294`: reward_mean=`10.130721762582738`, episode_len_mean=`568.0`
- iter `295`: reward_mean=`11.224680987465067`, episode_len_mean=`568.0`
- iter `296`: reward_mean=`12.670141301845494`, episode_len_mean=`568.0`
- iter `297`: reward_mean=`12.874215085654171`, episode_len_mean=`568.0`
- iter `298`: reward_mean=`13.339412000972745`, episode_len_mean=`568.0`
- iter `299`: reward_mean=`11.844333989772958`, episode_len_mean=`568.0`
