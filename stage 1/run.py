"""
CLI-точка входа для этапа 1 (LSTM-наукастинг).

Подкоманды:
  python "stage 1/run.py" catalog                    — статистика каталога
  python "stage 1/run.py" smoke [N]                  — скачать N событий, smoke-тест data.py
  python "stage 1/run.py" download                   — собрать все 5-мин профили
  python "stage 1/run.py" qc                         — пере-генерировать QC-график
  python "stage 1/run.py" prior                      — out-of-fold prior
  python "stage 1/run.py" train cv [fold]            — k-fold CV
  python "stage 1/run.py" train final                — финальная модель на SC23+24
  python "stage 1/run.py" evaluate baselines         — метрики бейзлайнов
  python "stage 1/run.py" evaluate full [ckpt]       — метрики LSTM + бейзлайнов
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def _usage():
    print(__doc__)
    sys.exit(0)


def main() -> None:
    if len(sys.argv) < 2:
        _usage()

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "catalog":
        from catalog import load_catalog, summary
        summary(load_catalog())

    elif cmd == "smoke":
        from data import smoke_test
        n = int(rest[0]) if rest else 3
        smoke_test(n)

    elif cmd == "download":
        from profiles import build_all, plot_qc
        idx = build_all()
        plot_qc(idx)

    elif cmd == "qc":
        import pandas as pd
        from profiles import plot_qc
        idx = pd.read_parquet(ROOT / "results" / "profiles_index.parquet")
        plot_qc(idx)

    elif cmd == "prior":
        from prior import compute_all
        compute_all()

    elif cmd == "train":
        sub = rest[0] if rest else "cv"
        if sub == "cv":
            from train import train_cv
            fold = int(rest[1]) if len(rest) > 1 else None
            train_cv(only_fold=fold)
        elif sub == "final":
            from train import train_final
            train_final()
        else:
            _usage()

    elif cmd == "evaluate":
        sub = rest[0] if rest else "baselines"
        if sub == "baselines":
            from evaluate import evaluate_split, summary_table, plot_delta_jmax_vs_time
            from evaluate import PLOTS_DIR
            df = evaluate_split("test", model_ckpt=None)
            print("\n== Сводка ==")
            print(summary_table(df).to_string(index=False))
            plot_delta_jmax_vs_time(df, PLOTS_DIR / "delta_jmax_vs_time_baselines.png")
        elif sub == "full":
            from evaluate import (evaluate_split, summary_table,
                                   plot_delta_jmax_vs_time, plot_case_studies, PLOTS_DIR)
            ckpt = Path(rest[1]) if len(rest) > 1 else \
                   ROOT / "results" / "checkpoints" / "final" / "final.pt"
            df = evaluate_split("test", model_ckpt=ckpt)
            print("\n== Сводка ==")
            print(summary_table(df).to_string(index=False))
            plot_delta_jmax_vs_time(df, PLOTS_DIR / "delta_jmax_vs_time.png")
            plot_case_studies(df)
        else:
            _usage()

    else:
        _usage()


if __name__ == "__main__":
    main()
