"""leg_succeeded: honest, state-delta task-completion (not proximity)."""
from task_chain import leg_succeeded

_BASE = {"carrying": None, "trash_remaining": 2,
         "drink_delivered": False, "package_delivered": False}


def s(**kw):
    d = dict(_BASE)
    d.update(kw)
    return d


def test_trash_success_requires_count_drop_not_proximity():
    assert leg_succeeded("collect_trash", s(trash_remaining=2),
                         s(trash_remaining=1), arrived=True)
    # arrived (near) but the env did not collect -> NOT success (the logged bug)
    assert not leg_succeeded("collect_trash", s(trash_remaining=2),
                             s(trash_remaining=2), arrived=True)


def test_fridge_success_is_drink_in_hand():
    assert leg_succeeded("go_to_fridge", s(carrying=None),
                         s(carrying="drink"), arrived=True)
    assert not leg_succeeded("go_to_fridge", s(carrying=None),
                             s(carrying=None), arrived=True)


def test_door_success_is_package_in_hand():
    assert leg_succeeded("go_to_door", s(carrying=None),
                         s(carrying="package"), arrived=True)
    assert not leg_succeeded("go_to_door", s(carrying=None),
                             s(carrying=None), arrived=True)


def test_human_success_is_delivery_when_carrying():
    assert leg_succeeded("go_to_human", s(carrying="drink"),
                         s(carrying=None, drink_delivered=True), arrived=True)
    # carrying but nothing delivered (just got near) -> not success
    assert not leg_succeeded("go_to_human", s(carrying="drink"),
                             s(carrying="drink"), arrived=True)


def test_human_with_empty_hands_falls_back_to_arrival():
    # pure navigation: no task effect possible, so success == arrived
    assert leg_succeeded("go_to_human", s(carrying=None), s(carrying=None),
                         arrived=True)
    assert not leg_succeeded("go_to_human", s(carrying=None), s(carrying=None),
                             arrived=False)
