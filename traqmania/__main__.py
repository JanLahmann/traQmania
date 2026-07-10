"""Entry point: `python -m traqmania` or the `traqmania` console script."""

from __future__ import annotations

import argparse

from traqmania.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="traqmania", description="traQmania QRL racing demo")
    parser.add_argument("--profile", default=None, help="config profile: pi4, pi5, exhibition, q6, q8, q10")
    parser.add_argument("--config", default=None, help="path to an extra TOML overlay")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(profile=args.profile, extra_path=args.config)
    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port

    import uvicorn

    from traqmania.server.app import create_app

    uvicorn.run(create_app(config), host=config["server"]["host"], port=config["server"]["port"])


if __name__ == "__main__":
    main()
