from planner.navigator_tool import NavigatorTool


def test_reset_returns_state_and_go_to_returns_outcome_shape():
    nav = NavigatorTool()  # loads the 314 champion
    assert nav.env.render_mode == "rgb_array"  # default stays headless
    s = nav.reset(seed=0)
    assert "robot_xy" in s
    out = nav.go_to("human")
    assert {"reached", "steps", "state"} <= set(out)
    assert isinstance(out["reached"], bool) and isinstance(out["steps"], int)


def test_unknown_destination_is_error_not_crash():
    nav = NavigatorTool()
    nav.reset(seed=0)
    out = nav.go_to("bogus")
    assert out["reached"] is False and "error" in out


def test_state_delegates_to_world():
    nav = NavigatorTool()
    nav.reset(seed=0)
    assert nav.state() == nav.world.state()
    assert "trash_remaining" in nav.state()


def test_trash_reached_implies_actually_collected():
    # Honesty contract: reached=True for trash must mean the env collected it
    # (trash_remaining dropped), not merely "within the loose 79px fixture
    # threshold". The env only picks trash up within ~31px; the harness reach
    # must match, or it declares success 1.5 tiles short with trash untouched.
    nav = NavigatorTool()
    saw_reach = False
    for seed in range(4):
        before = nav.reset(seed=seed)["trash_remaining"]
        out = nav.go_to("trash")
        if out["reached"]:
            saw_reach = True
            assert out["state"]["trash_remaining"] < before, (
                f"seed {seed}: reached=True but trash_remaining stayed {before}")
    assert saw_reach, "navigator never reached trash in 4 seeds; cannot verify"
