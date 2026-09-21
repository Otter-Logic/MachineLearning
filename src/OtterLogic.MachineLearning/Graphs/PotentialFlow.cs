namespace OtterLogic.MachineLearning.Graphs;

/// <summary>
/// The potential at every node of a <see cref="WeightedGraph"/> once something is
/// injected at some nodes and drained at others, and so the flow along every edge.
/// </summary>
/// <param name="Potential">
/// Potential at each node, zero at a grounded node. NaN where no grounded node can be
/// reached, because nothing injected there has anywhere to go.
/// </param>
/// <param name="Reached">Whether a grounded node can be reached from each node.</param>
/// <param name="Iterations">Conjugate-gradient iterations taken, summed over the graph's pieces.</param>
/// <param name="Converged">False when any piece stopped at the iteration cap short of the tolerance.</param>
public sealed record PotentialFlowResult(double[] Potential, bool[] Reached, int Iterations, bool Converged)
{
    /// <summary>
    /// Flow along an edge of weight <paramref name="weight"/> from <paramref name="from"/>
    /// to <paramref name="to"/>: positive when it runs that way. Zero when either end is unreached.
    /// </summary>
    public double Flow(int from, int to, double weight)
        => Reached[from] && Reached[to] ? weight * (Potential[from] - Potential[to]) : 0.0;
}

/// <summary>
/// Flow through a graph from where something enters to where it drains, every edge
/// carrying it in proportion to its weight and the drop in potential along it — the
/// current in a resistor network, heat in a conductor, a random walk absorbed at the
/// grounded nodes.
/// <para>
/// A shortest route answers "which way is nearest"; this answers "how much passes
/// through here", and it answers it without choosing. Where two routes are as good
/// as each other the flow divides between them, so a symmetric graph gets a
/// symmetric answer — which a route search cannot give, since it must break the tie
/// one way or the other, and then two nodes that are mirror images of each other
/// come out different.
/// </para>
/// <para>
/// Solves <c>L φ = b</c> on the ungrounded nodes, <c>L</c> the graph Laplacian and
/// <c>b</c> the injection, by conjugate gradients preconditioned with the degree.
/// The reduced Laplacian is positive definite in any piece of the graph holding a
/// grounded node, so the solve cannot fail on a graph that is a mechanism, a tree or
/// anything else a stiffness matrix would refuse; a piece with no grounded node is
/// reported unreached rather than solved. Nothing is random and every sum runs in
/// neighbour order, so the same graph gives the same bits.
/// </para>
/// </summary>
public static class PotentialFlow
{
    /// <summary>Solves for the potentials.</summary>
    /// <param name="graph">Edge weights are conductances: a heavier edge carries more for the same drop.</param>
    /// <param name="injection">What enters at each node; negative draws off. Ignored at grounded and unreached nodes.</param>
    /// <param name="grounded">Nodes held at zero potential, where everything drains. Duplicates are ignored.</param>
    /// <param name="tolerance">Residual, relative to the injection's size, at which the solve stops.</param>
    public static PotentialFlowResult Solve(
        WeightedGraph graph, IReadOnlyList<double> injection, IEnumerable<int> grounded, double tolerance = 1e-10)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(injection);
        ArgumentNullException.ThrowIfNull(grounded);

        int n = graph.NodeCount;
        if (injection.Count != n)
            throw new ArgumentException($"The graph has {n} nodes but {injection.Count} injections were given.", nameof(injection));
        if (!(tolerance > 0.0) || !double.IsFinite(tolerance))
            throw new ArgumentOutOfRangeException(nameof(tolerance), tolerance, "Tolerance must be above zero.");

        for (int i = 0; i < n; i++)
            if (!double.IsFinite(injection[i]))
                throw new ArgumentException($"Injection {i} is {injection[i]}; injections must be finite.", nameof(injection));

        var isGrounded = new bool[n];
        foreach (int g in grounded)
        {
            if (g < 0 || g >= n)
                throw new ArgumentOutOfRangeException(nameof(grounded), g, $"Grounded node {g} is outside 0..{n - 1}.");
            isGrounded[g] = true;
        }

        var reached = Reach(graph, isGrounded);
        var potential = new double[n];
        var degree = new double[n];
        var b = new double[n];
        bool any = false;

        for (int i = 0; i < n; i++)
        {
            if (!reached[i])
            {
                potential[i] = double.NaN;
                continue;
            }

            if (isGrounded[i])
                continue;

            degree[i] = graph.Degree(i);
            b[i] = injection[i];
            any |= b[i] != 0.0;
        }

        if (!any)
            return new PotentialFlowResult(potential, reached, 0, true);

        bool Free(int i) => reached[i] && !isGrounded[i];

        void Apply(double[] x, double[] y)
        {
            for (int i = 0; i < n; i++)
            {
                if (!Free(i))
                    continue;

                var neighbours = graph.Neighbours(i);
                var weights = graph.EdgeWeights(i);
                double sum = degree[i] * x[i];
                for (int e = 0; e < neighbours.Length; e++)
                    if (Free(neighbours[e]))
                        sum -= weights[e] * x[neighbours[e]];

                y[i] = sum;
            }
        }

        double Dot(double[] u, double[] v)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                if (Free(i))
                    sum += u[i] * v[i];
            return sum;
        }

        var x = new double[n];
        var r = (double[])b.Clone();
        var z = new double[n];
        var p = new double[n];
        var ap = new double[n];

        for (int i = 0; i < n; i++)
            if (Free(i))
                p[i] = z[i] = r[i] / degree[i];

        double target = tolerance * tolerance * Dot(b, b);
        double rz = Dot(r, z);
        int cap = 10 * n + 100;
        int iterations = 0;
        bool converged = Dot(r, r) <= target;

        while (!converged && iterations < cap)
        {
            Apply(p, ap);
            double curvature = Dot(p, ap);
            if (!(curvature > 0.0))
                break;

            double step = rz / curvature;
            for (int i = 0; i < n; i++)
            {
                if (!Free(i))
                    continue;
                x[i] += step * p[i];
                r[i] -= step * ap[i];
            }

            iterations++;
            if (Dot(r, r) <= target)
            {
                converged = true;
                break;
            }

            for (int i = 0; i < n; i++)
                if (Free(i))
                    z[i] = r[i] / degree[i];

            double next = Dot(r, z);
            double keep = next / rz;
            rz = next;

            for (int i = 0; i < n; i++)
                if (Free(i))
                    p[i] = z[i] + keep * p[i];
        }

        for (int i = 0; i < n; i++)
            if (Free(i))
                potential[i] = x[i];

        return new PotentialFlowResult(potential, reached, iterations, converged);
    }

    /// <summary>Every node some grounded node can be reached from.</summary>
    private static bool[] Reach(WeightedGraph graph, bool[] grounded)
    {
        var reached = new bool[graph.NodeCount];
        var stack = new Stack<int>();

        for (int g = 0; g < grounded.Length; g++)
        {
            if (!grounded[g] || reached[g])
                continue;

            reached[g] = true;
            stack.Push(g);
            while (stack.Count > 0)
                foreach (int next in graph.Neighbours(stack.Pop()))
                    if (!reached[next])
                    {
                        reached[next] = true;
                        stack.Push(next);
                    }
        }

        return reached;
    }
}
