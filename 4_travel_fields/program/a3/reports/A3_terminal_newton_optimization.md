# A3 Terminal Event Optimization: Polynomial Newton vs Repeated GL5 Bisection

**Status:** PASS

## Configuration

- Cases: `500`
- Octaves: `5`
- Terminal reference: 28-step GL5 bisection
- Candidate: one combined degree-9 density build, degree-10 optical-depth polynomial, safeguarded Newton

## Accuracy

- P99 distance error: `1.330680e-10`
- Maximum distance error: `1.463958e-10`

## Iteration behavior

- Mean Newton iterations: `3.040`
- P95 Newton iterations: `4.0`
- Maximum Newton iterations: `4`
- Mean safeguarded fallbacks: `0.000`
- Maximum fallbacks: `0`

## Terminal-cost proxy

Repeated GL5 bisection:

- Mean scalar octave-density evaluations: `700.0`

Polynomial Newton:

- Mean octave polynomial builds: `5.0`

Treating one octave polynomial build as one heavy unit, the terminal-stage work
drops by approximately `99.3%`
relative to repeated GL5 density evaluations.

## Consequence

The image benchmark used repeated GL5 bisection and therefore substantially
overstated A3 terminal-event cost.  The intended A3 implementation should:

1. consume complete common segments with exact GL5;
2. build the combined degree-9 density only once in the terminal segment;
3. integrate to a degree-10 optical-depth polynomial;
4. solve with safeguarded Newton.

This does not make A3 a production replacement for ray marching, but it
restores the theoretically intended architecture and removes its largest
avoidable event-location overhead.
