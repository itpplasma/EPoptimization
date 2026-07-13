from evaluate_spatial_surface import parser


def test_parser_accepts_scheduler_timeout() -> None:
    args = parser().parse_args(
        [
            "--wout",
            "wout.nc",
            "--design",
            "design",
            "--out",
            "output",
            "--simple-executable",
            "simple.x",
            "--timeout",
            "7200",
        ]
    )

    assert args.timeout == 7200.0
