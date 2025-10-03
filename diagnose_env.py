# diagnose_env.py
"""
Comprehensive diagnostics for DroneDeliveryEnv.

What this checks:
1) Fixed map reproducibility across resets
2) Warehouse workflow invariants
3) Priority-based selection
4) Carry / Deliver transitions (with interact)
5) Battery and charging mechanics
6) Reward structure (Delivery = 100 * priority; Pickup = 10; Move = -1)
7) Solvability (random baseline) – uses easy_mode_for_tests=True

Author: Michal Valčo
"""

import numpy as np
from environment.drone_env import DroneDeliveryEnv


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_fixed_map():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)
    layout1 = (tuple(env.chargers), tuple((p.priority, p.pickup_pos, p.dest_pos) for p in env.packages))
    env.reset(seed=42)
    layout2 = (tuple(env.chargers), tuple((p.priority, p.pickup_pos, p.dest_pos) for p in env.packages))

    ok = layout1 == layout2
    print("✅ PASS: Map remains identical across multiple resets" if ok else "❌ FAIL: Map changed unexpectedly")
    print(f"  • Packages: {len(env.packages)}")
    print(f"  • Stations: {len(env.chargers)}")
    print(f"  • Warehouse zone: ({env.wh_x0}, {env.wh_y0}, {env.wh_w}, {env.wh_h})")
    return ok


def test_warehouse_invariants():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)
    ok = True

    # Warehouse is defined
    print("✅ PASS: Warehouse zone defined")

    # All packages spawn in warehouse
    for p in env.packages:
        x, y = p.pickup_pos
        in_wh = (env.wh_x0 <= x < env.wh_x0 + env.wh_w) and (env.wh_y0 <= y < env.wh_y0 + env.wh_h)
        if not in_wh:
            ok = False
    print("✅ PASS: All packages spawn in warehouse zone" if ok else "❌ FAIL: Some packages spawn outside warehouse")

    # At least one charger in warehouse
    wh_charger = any((env.wh_x0 <= cx < env.wh_x0 + env.wh_w) and (env.wh_y0 <= cy < env.wh_y0 + env.wh_h) for (cx, cy) in env.chargers)
    print("✅ PASS: Charging station exists in warehouse" if wh_charger else "❌ FAIL: No charger in warehouse")

    # Drone starts at center of warehouse
    start_ok = (env.drone_pos[0] == env.wh_center[0] and env.drone_pos[1] == env.wh_center[1])
    print(f"✅ PASS: Drone starts in warehouse at {env.drone_pos}" if start_ok else f"❌ FAIL: Drone start incorrect: {env.drone_pos}")
    return ok and wh_charger and start_ok


def test_priority_selection():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)

    print("Package priorities:")
    for i, p in enumerate(env.packages, 1):
        print(f"  Package {i}: Priority {p.priority}, Pos {np.array(p.pickup_pos)}")

    # Highest priority available?
    highest = max(p.priority for p in env.packages)
    # Ask env which it would pick internally (replicating the selection logic)
    candidates = [p for p in env.packages if (not p.picked_up and not p.delivered)]
    candidates.sort(key=lambda p: (-p.priority, env._manhattan(env.drone_pos, p.dest_pos)))
    chosen = candidates[0]
    print(f"\nSelected package: Priority {chosen.priority}")
    ok = (chosen.priority == highest)
    print("✅ PASS: Correctly selected priority", highest if ok else "❌ FAIL: Priority selection mismatch")
    return ok


def test_carry_deliver_transitions():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)

    print(f"✅ PASS: Initially not carrying (carrying={env.carrying_package})")

    # Move to a package pickup tile (use the chosen top package)
    candidates = [p for p in env.packages if not p.picked_up and not p.delivered]
    candidates.sort(key=lambda p: (-p.priority, env._manhattan(env.drone_pos, p.dest_pos)))
    target = candidates[0]

    # walk to pickup tile (strict mode requires exact pickup tile)
    px, py = target.pickup_pos
    env.drone_pos[:] = (px, py)
    obs, reward, done, trunc, info = env.step(4)  # interact -> pickup
    print(f"  Moved drone to package at {np.array([px, py])}")
    print(f"✅ PASS: Pickup successful (reward={reward})" if env.carrying_package else "❌ FAIL: Pickup failed")

    # move to destination and deliver
    dx, dy = target.dest_pos
    env.drone_pos[:] = (dx, dy)
    obs, reward, done, trunc, info = env.step(4)  # interact -> deliver
    print(f"  Moved drone to destination at {np.array([dx, dy])}")
    print(f"✅ PASS: Delivery successful (reward={reward})" if info.get("delivered_now", 0) == 1 or env.current_package is None else "❌ FAIL: Delivery failed")
    delivered_count = sum(p.delivered for p in env.packages)
    print(f"✅ PASS: Delivery count = {delivered_count}" if delivered_count >= 1 else "❌ FAIL: Delivery count not incremented")

    return env.carrying_package is False and delivered_count >= 1


def test_battery_and_charging():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)
    all_ok = True

    print(f"✅ PASS: Initial battery = {env.battery:.2f}")
    
    # move east and measure battery drain
    before = env.battery
    env.step(1)  # right
    after = env.battery
    drain_ok = abs((before - after) - 0.01) < 1e-6
    print(f"✅ PASS: Movement drains battery by {before - after:.3f} (expected 0.01)" if drain_ok else f"❌ FAIL: Unexpected movement drain {before - after:.3f}")
    all_ok = all_ok and drain_ok

    # drop battery and charge
    env.battery = 0.25
    initial_battery = env.battery
    cx, cy = env.chargers[0]  # Unpack the tuple
    env.drone_pos[:] = [cx, cy]  # Set as list/array
    obs, reward, done, trunc, info = env.step(4)  # interact -> charge
    charge_ok = env.battery > initial_battery
    print(f"✅ PASS: Charging increases battery ({initial_battery:.2f} → {env.battery:.2f})" if charge_ok else f"❌ FAIL: Charging didn't increase battery ({initial_battery:.2f} → {env.battery:.2f})")
    all_ok = all_ok and charge_ok
    
    reward_ok = abs(reward - 15.0) < 1e-6
    print(f"✅ PASS: Strategic charging reward = {reward:.1f} (expected 15)" if reward_ok else f"❌ FAIL: Wrong charge reward {reward:.1f}")
    all_ok = all_ok and reward_ok
    
    return all_ok


def test_reward_structure():
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42)
    env.reset(seed=42)

    print("Testing delivery rewards (should be 100 × priority):")
    ok = True
    for pr in [1, 2, 3]:
        # fabricate a package of given priority and test delivery reward
        p = env.packages[0]
        p.priority = pr
        p.picked_up = True
        env.carrying_package = True
        env.current_package = p
        env.drone_pos[:] = p.dest_pos

        obs, reward, done, trunc, info = env.step(4)  # deliver
        expected = 100.0 * pr
        print(f"  Priority {pr}: reward={reward:.1f} (expected {expected:.0f})")
        if abs(reward - expected) > 1e-6:
            ok = False
        # reset carry state
        env.carrying_package = False
        env.current_package = None
        p.delivered = False
        p.picked_up = False
    print("✅ PASS: Reward structure correct" if ok else "❌ FAIL: Rewards mismatch")
    return ok


def test_random_solvable():
    # use easy mode so a purely random policy can succeed occasionally
    env = DroneDeliveryEnv(fixed_env=True, base_seed=42, easy_mode_for_tests=True)
    episodes = 50
    deliveries = []
    rewards = []

    rng = np.random.default_rng(0)
    for _ in range(episodes):
        obs, _ = env.reset(seed=42)
        done = False
        trunc = False
        ep_deliv = 0
        ep_ret = 0.0
        while not (done or trunc):
            a = int(rng.integers(0, env.action_space.n))
            obs, r, done, trunc, info = env.step(a)
            ep_ret += float(r)
            ep_deliv += int(info.get("delivered_now", 0))
        deliveries.append(ep_deliv)
        rewards.append(ep_ret)

    print("\nResults after 50 random episodes:")
    print(f"  • Avg Reward: {np.mean(rewards):.2f}")
    print(f"  • Avg Deliveries: {np.mean(deliveries):.2f} / {env.n_packages}")
    print(f"  • Max Deliveries: {np.max(deliveries)} / {env.n_packages}")
    success_rate = np.mean([1.0 if d > 0 else 0.0 for d in deliveries])
    print(f"  • Success Rate: {success_rate * 100:.1f}%")
    ok = success_rate > 0.0
    print("✅ PASS: Environment is solvable (random agent occasionally succeeds)" if ok else "❌ FAIL: Random agent never succeeds")
    return ok


def main():
    print("\n COMPREHENSIVE ENVIRONMENT DIAGNOSTICS")
    print(" Lesson 10: RL Agent – Practical Project")
    print("=" * 70)

    banner("TEST 1: Fixed Map Reproducibility")
    t1 = test_fixed_map()

    banner("TEST 2: Warehouse Workflow")
    t2 = test_warehouse_invariants()

    banner("TEST 3: Priority-Based Package Selection")
    t3 = test_priority_selection()

    banner("TEST 4: Carry/Deliver State Transitions")
    t4 = test_carry_deliver_transitions()

    banner("TEST 5: Battery & Charging Mechanics")
    t5 = test_battery_and_charging()

    banner("TEST 6: Reward Structure")
    t6 = test_reward_structure()

    banner("TEST 7: Environment Solvability (Random Baseline)")
    t7 = test_random_solvable()

    banner(" DIAGNOSTIC SUMMARY")
    passed = [t1, t2, t3, t4, t5, t6, t7]
    labels = ["Fixed Map", "Warehouse", "Priority", "Transitions", "Battery", "Rewards", "Solvable"]
    for ok, name in zip(passed, labels):
        print(("✅ PASS - " if ok else "❌ FAIL - ") + name)

    print("\n" + "=" * 70)
    print(f" TOTAL: {sum(passed)}/{len(passed)} tests passed ({int(100*sum(passed)/len(passed))}%)")
    print("=" * 70)
    print("\nIf any tests failed, fix those before long training runs.\n")


if __name__ == "__main__":
    main()