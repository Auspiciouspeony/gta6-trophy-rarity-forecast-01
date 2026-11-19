# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Auspiciouspeony/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Auspiciouspeony/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate a week after launch, as five quantiles a trophy, from the house's recorded reads of the trophy page. The forecaster is a refresh-aware logit-space trend fit: Exophase refreshes a game's counts hours apart, so only the reads where something changed carry signal; each trophy's logit rate is fit against log time on the recent fresh reads, shrunk toward its group's slope, and spread by a calibrated logit width. A page that has never refreshed gets a dilution adjustment for the overdue first refresh. Scored by pinball loss on the logit scale; tuned on a launch simulator against the house baselines.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is standard library only (Python 3.13), deterministic, and reads every `*.in` case in the input directory, writing one `*.out` forecast per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wa0stw5e&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
