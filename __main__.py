"""
Главная точка входа в проект прогнозирования СПС.

Запуск без аргументов — интерактивное меню:
  python __main__.py

Прямой вызов:
  python __main__.py data process
  python __main__.py data pdf
  python __main__.py data sinp
  python __main__.py data noaa
  python __main__.py data fluence
  python __main__.py train regression
  python __main__.py train probabilistic
  python __main__.py train classification
  python __main__.py compare regression
  python __main__.py compare probabilistic
  python __main__.py compare classification
  python __main__.py tune regression
  python __main__.py tune probabilistic
  python __main__.py tune classification
  python __main__.py importance regression
  python __main__.py importance probabilistic
  python __main__.py importance classification
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# ── ANSI-цвета (работают на Windows 10+ / любом терминале с поддержкой) ───────
_C = {
    "bold":   "\033[1m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}

def _c(text, *codes):
    return "".join(_C[c] for c in codes) + text + _C["reset"]


# ── Реестр действий ────────────────────────────────────────────────────────────

ACTIONS = {
    # (group, key): (description, script_path)
    ("data", "process"):  ("Обработать каталог → processed.csv",
                           "data/processing.py"),
    ("data", "pdf"):      ("Парсинг PDF каталога 23-го цикла (Логачёв 2016)",
                           "data/parce_pdf.py"),
    ("data", "sinp"):     ("Парсинг HTML СИНП МГУ (циклы 24–25)",
                           "data/parce_sinp.py"),
    ("data", "noaa"):     ("Обогащение точными временами вспышек (NOAA)",
                           "data/noaa.py"),
    ("data", "fluence"):  ("Сбор флюэнса рентген. вспышек (GOES XRS, SunPy)",
                           "data/goes_fluence.py"),

    ("train", "regression"):     ("Точечный прогноз Jmax и T_delta (RMSLE)",
                                  "pipelines/regression/train.py"),
    ("train", "probabilistic"):  ("Интервальный прогноз 80% CI (Winkler score)",
                                  "pipelines/probabilistic/train.py"),
    ("train", "classification"): ("Классификация по NOAA S-классам (LogLoss)",
                                  "pipelines/classification/train.py"),

    ("compare", "regression"):     ("Сравнение наборов признаков — регрессия",
                                    "pipelines/regression/compare.py"),
    ("compare", "probabilistic"):  ("Сравнение наборов признаков — интервальный",
                                    "pipelines/probabilistic/compare.py"),
    ("compare", "classification"): ("Сравнение наборов признаков — классификация",
                                    "pipelines/classification/compare.py"),
    ("compare", "tdelta_clf"):     ("Сравнение наборов признаков — классификация T_delta",
                                    "pipelines/tdelta_clf/compare.py"),

    ("tune", "regression"):     ("Тюнинг гиперпараметров — регрессия",
                                 "pipelines/regression/tune.py"),
    ("tune", "probabilistic"):  ("Тюнинг гиперпараметров — интервальный",
                                 "pipelines/probabilistic/tune.py"),
    ("tune", "classification"): ("Тюнинг гиперпараметров — классификация",
                                 "pipelines/classification/tune.py"),

    ("importance", "regression"):     ("Вклад признаков — регрессия (builtin/SHAP/MI)",
                                       "pipelines/regression/importance.py"),
    ("importance", "probabilistic"):  ("Вклад признаков — интервальный прогноз",
                                       "pipelines/probabilistic/importance.py"),
    ("importance", "classification"): ("Вклад признаков — классификация",
                                       "pipelines/classification/importance.py"),
}

GROUPS = [
    ("data",    "Данные",     ["process", "pdf", "sinp", "noaa", "fluence"]),
    ("train",   "Обучение",   ["regression", "probabilistic", "classification"]),
    ("compare", "Сравнение",  ["regression", "probabilistic", "classification", "tdelta_clf"]),
    ("tune",      "Тюнинг",           ["regression", "probabilistic", "classification"]),
    ("importance","Вклад признаков",  ["regression", "probabilistic", "classification"]),
]


# ── Выполнение ─────────────────────────────────────────────────────────────────

def run_action(group: str, key: str) -> int:
    action = ACTIONS.get((group, key))
    if not action:
        print(_c(f"Неизвестное действие: {group} {key}", "red"))
        return 1

    desc, script = action
    script_path = ROOT / script

    if not script_path.exists():
        print(_c(f"Скрипт не найден: {script_path}", "red"))
        return 1

    print()
    print(_c(f"► {desc}", "bold", "cyan"))
    print(_c(f"  {script}", "dim"))
    print(_c("─" * 55, "dim"))

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.returncode


# ── Интерактивное меню ─────────────────────────────────────────────────────────

def _print_menu():
    print()
    print(_c("╔══════════════════════════════════════════════╗", "cyan"))
    print(_c("║   Прогнозирование СПС — Главное меню         ║", "cyan", "bold"))
    print(_c("╚══════════════════════════════════════════════╝", "cyan"))

    idx = 1
    item_map = {}  # номер → (group, key)

    for group, group_label, keys in GROUPS:
        print()
        print(_c(f"  {group_label}", "bold", "yellow"))
        for key in keys:
            desc, script = ACTIONS[(group, key)]
            print(f"    {_c(str(idx), 'green')}  {desc}")
            print(_c(f"       {script}", "dim"))
            item_map[idx] = (group, key)
            idx += 1

    print()
    print(f"    {_c('0', 'red')}  Выход")
    print()
    return item_map


def interactive_menu():
    while True:
        item_map = _print_menu()
        try:
            raw = input(_c("Введите номер действия: ", "bold")).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if raw == "0" or raw.lower() in ("q", "exit", "quit"):
            break

        try:
            choice = int(raw)
        except ValueError:
            print(_c("  Введите число из списка.", "red"))
            continue

        if choice not in item_map:
            print(_c("  Нет такого пункта.", "red"))
            continue

        group, key = item_map[choice]
        rc = run_action(group, key)

        print()
        if rc == 0:
            print(_c("  ✓ Завершено успешно.", "green", "bold"))
        else:
            print(_c(f"  ✗ Завершено с кодом {rc}.", "red"))

        try:
            input(_c("\n  Нажмите Enter для возврата в меню...", "dim"))
        except (KeyboardInterrupt, EOFError):
            break


# ── argparse ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python __main__.py",
        description="Прогнозирование параметров СПС",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_help_epilog(),
    )
    sub = parser.add_subparsers(dest="group", metavar="ГРУППА")

    for group, group_label, keys in GROUPS:
        sp = sub.add_parser(group, help=group_label)
        sp.add_argument("action", choices=keys, metavar="ДЕЙСТВИЕ",
                        help=" | ".join(keys))

    return parser


def _build_help_epilog() -> str:
    lines = ["\nДоступные команды:"]
    for group, group_label, keys in GROUPS:
        lines.append(f"\n  {group_label}:")
        for key in keys:
            desc, _ = ACTIONS[(group, key)]
            lines.append(f"    {group} {key:<18} {desc}")
    return "\n".join(lines).encode("ascii", errors="replace").decode("ascii")


# ── Точка входа ────────────────────────────────────────────────────────────────

def main():
    # Включаем ANSI на Windows + UTF-8 для консоли
    if sys.platform == "win32":
        import os
        os.system("")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # ── GPU-детектирование ──────────────────────────────────────────────────
    try:
        import torch, os as _os
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            print(_c(f"GPU: {gpu.name}  ({gpu.total_memory/1e9:.1f} GB VRAM)", "green", "bold"))
            _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        else:
            print(_c("GPU: не обнаружена -- используется CPU (n_jobs=-1)", "yellow"))
    except ImportError:
        print(_c("GPU: torch не установлен", "yellow"))

    if len(sys.argv) == 1:
        # Нет аргументов → интерактивное меню
        interactive_menu()
        return

    parser = build_parser()
    args = parser.parse_args()

    if not args.group:
        parser.print_help()
        return

    rc = run_action(args.group, args.action)
    sys.exit(rc)


if __name__ == "__main__":
    main()
