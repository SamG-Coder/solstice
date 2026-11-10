"""Lays out the coastal circuit, and writes its `.track` file.

    python tools/track/make_coast.py [scenes/circuit/coast.track]

The circuit is drawn here as a handful of points the centre line passes through. A smooth closed
curve is run through them, measured, and written out as a point every three metres, each with
the way across the road there, its width, and what lies either side: kerbs through the corners,
gravel outside the ones a car arrives at fastest, grass elsewhere.

From that one file the bake builds the road, its kerbs and lines, the land shaped to it, the
barriers and the planting (apps/bake/src/circuit.rs), and the game reads what is under each
tyre and times laps (crates/solstice-game/src/track.rs).

East (+x) is the sea. North is -z. The lap runs clockwise seen from above: north up the start
straight on the west side, east over the rise, south along the cliff top, and back west through
the meadow.
"""

import math
import sys

NAME = "Kestrel Head"
STEP = 3.0
HALF_WIDTH = 6.5
# How far the whole circuit is raised above the heights below: its cliffs are this much higher.
LIFT = 12.0
# The centre line's points: east, south, height above the sea, in metres.
POINTS = [
    (-300.0, 60.0, 18.0),     # the start line, half-way up the straight
    (-300.0, -120.0, 19.0),
    (-300.0, -300.0, 21.0),   # braking
    (-280.0, -392.0, 23.0),   # 1: a right-hander, climbing
    (-200.0, -440.0, 25.0),
    (-70.0, -436.0, 26.5),    # the crest
    (60.0, -378.0, 25.0),     # 2 and 3: the esses, falling
    (190.0, -432.0, 22.5),
    (318.0, -410.0, 19.5),
    (424.0, -322.0, 17.0),    # 4: a fast right, out onto the cliff top
    (468.0, -176.0, 15.5),
    (452.0, -24.0, 15.0),     # 5 and 6: kinks along the cliff, flat out
    (480.0, 126.0, 14.0),
    (478.0, 262.0, 12.5),     # braking, downhill
    (450.0, 350.0, 11.5),     # 7: the hairpin at the point
    (378.0, 368.0, 12.0),
    (290.0, 322.0, 14.5),
    (176.0, 388.0, 18.0),     # 8 and 9: left, then right, climbing through the meadow
    (48.0, 350.0, 20.5),
    (-80.0, 404.0, 21.0),
    (-206.0, 400.0, 20.0),    # 10: the last corner, a right onto the straight
    (-284.0, 326.0, 19.0),
    (-300.0, 210.0, 18.2),
]


def catmull_rom(points, per=60):
    """A closed centripetal Catmull-Rom curve through the points, densely sampled."""
    n = len(points)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k - 1) % n] for k in range(4))

        def knot(t, a, b):
            return t + max(math.dist(a[:2], b[:2]), 1e-6) ** 0.5

        t0 = 0.0
        t1, t2 = knot(t0, p0, p1), 0.0
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        for step in range(per):
            t = t1 + (t2 - t1) * step / per

            def lerp(a, b, ta, tb):
                w = (t - ta) / (tb - ta)
                return tuple(x + (y - x) * w for x, y in zip(a, b))

            a1, a2, a3 = lerp(p0, p1, t0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
            b1, b2 = lerp(a1, a2, t0, t2), lerp(a2, a3, t1, t3)
            out.append(lerp(b1, b2, t1, t2))
    return out


def resample(dense, step):
    """Points a fixed way apart along a closed curve: as near `step` as divides the lap evenly."""
    lengths = [0.0]
    for a, b in zip(dense, dense[1:] + dense[:1]):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1]
    count = round(total / step)
    out, j = [], 0
    for k in range(count):
        at = total * k / count
        while lengths[j + 1] < at:
            j += 1
        a, b = dense[j], dense[(j + 1) % len(dense)]
        w = (at - lengths[j]) / (lengths[j + 1] - lengths[j])
        out.append(tuple(x + (y - x) * w for x, y in zip(a, b)))
    return out, total


def smooth(values, passes, closed=True):
    n = len(values)
    for _ in range(passes):
        values = [(values[i - 1] + 2 * values[i] + values[(i + 1) % n]) / 4 for i in range(n)]
    return values


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scenes/circuit/coast.track"
    # x east, z south, y up: as the engine has them.
    samples, length = resample([(x, y + LIFT, z) for x, z, y in catmull_rom(POINTS)], STEP)
    n = len(samples)
    step = length / n

    # How sharply the line turns at each point, per metre: above zero to the left.
    turn = []
    for i in range(n):
        ax, _, az = samples[i - 1]
        bx, _, bz = samples[i]
        cx, _, cz = samples[(i + 1) % n]
        f1 = (bx - ax, bz - az)
        f2 = (cx - bx, cz - bz)
        cross = f1[1] * f2[0] - f1[0] * f2[1]  # y of f1 x f2, with y up
        turn.append(math.atan2(cross, f1[0] * f2[0] + f1[1] * f2[1]) / step)
    turn = smooth(turn, 6)

    lines = [
        "# Written by tools/track/make_coast.py: change the circuit there, not here.",
        f"name {NAME}",
        f"sector {n // 3}",
        f"sector {2 * n // 3}",
    ]
    # Gravel outside the slow corners and on the way into them. A trap comes in and goes out
    # over a couple of dozen metres, as the barrier behind it does.
    gravel = {"l": [0.0] * n, "r": [0.0] * n}
    for i in range(n):
        radius = 1.0 / max(abs(turn[i]), 1e-6)
        # Where a car that brakes too late for a slow corner ends up: straight on, on the
        # outside, from a little before the corner to its end.
        soon = min(range(0, 20), key=lambda k: 1.0 / max(abs(turn[(i + k) % n]), 1e-6))
        ahead = 1.0 / max(abs(turn[(i + soon) % n]), 1e-6)
        if ahead < 86.0:
            outside = "r" if turn[(i + soon) % n] > 0 else "l"
            if radius < 86.0:
                outside = "r" if turn[i] > 0 else "l"
            gravel[outside][i] = 18.0
    for side in gravel:
        gravel[side] = smooth(gravel[side], 14)

    corners, in_corner, tightest = [], False, 1e9
    for i in range(n):
        x, y, z = samples[i]
        px, py, pz = samples[i - 1]
        nx, ny, nz = samples[(i + 1) % n]
        fx, fy, fz = nx - px, ny - py, nz - pz
        flat = math.hypot(fx, fz)
        fx, fy, fz = fx / flat, fy / flat, fz / flat
        # Across to the left, level: up crossed with the way ahead.
        lx, lz = fz, -fx
        radius = 1.0 / max(abs(turn[i]), 1e-6)
        tightest = min(tightest, radius)
        # Banked into each corner, more the tighter it is, to four degrees at most.
        bank = math.radians(min(450.0 / radius, 4.0)) * (1 if turn[i] > 0 else -1)
        left = (lx * math.cos(bank), -math.sin(bank), lz * math.cos(bank))
        # The start straight is wider, for the grid.
        along = i * step
        from_start = min(along, length - along)
        half = HALF_WIDTH + 1.0 * max(0.0, 1.0 - from_start / 180.0)

        cornering = radius < 260.0
        if cornering and not in_corner:
            corners.append([i, i, radius])
        if cornering:
            corners[-1][1] = i
            corners[-1][2] = min(corners[-1][2], radius)
        in_corner = cornering
        # Kerbs through the corners, and wherever there is a trap behind them; a strip of
        # concrete along both sides of the start straight.
        sides = {"l": [], "r": []}
        for side in sides:
            trap = gravel[side][i]
            if radius < 260.0 or trap > 0.6:
                sides[side].append("kerb:1.4")
            if trap > 0.6:
                sides[side].append(f"gravel:{trap:.1f}")
        if from_start < 150.0:
            sides["l"].append("concrete:4")
            sides["r"].append("concrete:4")
        bands = "".join(f" {k}={','.join(v)}" for k, v in sides.items() if v)
        lines.append(
            f"p {x:.3f} {y:.3f} {z:.3f} {left[0]:.5f} {left[1]:.5f} {left[2]:.5f} {half:.2f} {half:.2f}{bands}"
        )
    with open(out, "w", newline="\n") as file:
        file.write("\n".join(lines) + "\n")

    print(f"{NAME}: {length:.0f} m, {n} points, tightest radius {tightest:.0f} m")
    for first, last, radius in corners:
        side = "left" if turn[(first + last) // 2] > 0 else "right"
        print(f"  corner {side:5s} from {first * step:5.0f} m to {last * step:5.0f} m, radius {radius:4.0f} m")
    # Where to put a car on the grid: 30 m before the line, facing along the lap.
    gx, gy, gz = samples[-round(30.0 / step)]
    ax, _, az = samples[-round(30.0 / step) + 1]
    yaw = math.degrees(math.atan2(-(ax - gx), -(az - gz)))
    print(f"  grid: actor ... {gx:.2f} {gy + 0.05:.2f} {gz:.2f} {yaw:.1f}")


main()
