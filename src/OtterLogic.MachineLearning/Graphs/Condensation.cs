namespace OtterLogic.MachineLearning.Graphs;

/// <summary>
/// A directed graph with its cycles folded away: which nodes depend on each other in
/// a loop, and how long a chain of dependence hangs below each.
/// </summary>
/// <param name="Component">
/// Strongly connected component of each node — nodes that can each reach the other
/// along the arcs. Numbered in order of their lowest node.
/// </param>
/// <param name="ComponentCount">Number of components. Equal to the node count when there is no cycle.</param>
/// <param name="Height">
/// Per component, the longest chain of arcs leading out of it, counted in components:
/// zero for one with no arc out to another.
/// </param>
public sealed record CondensationResult(int[] Component, int ComponentCount, int[] Height)
{
    /// <summary>The height of the component <paramref name="node"/> belongs to.</summary>
    public int HeightOf(int node) => Height[Component[node]];
}

/// <summary>
/// Orders the nodes of a directed graph by what they depend on, treating every
/// cycle as one thing.
/// <para>
/// An arc says its tail depends on its head. Dependence that runs one way gives a
/// hierarchy, and a node's <see cref="CondensationResult.Height"/> is its rank in it.
/// Dependence that runs both ways — a cycle — is not a hierarchy at all, and the
/// honest reading is that everything on the cycle shares a rank; forcing an order
/// onto it would only record which arc the search happened to meet first. So the
/// cycles are folded into single components first, which always leaves an acyclic
/// graph, and the ranks are read from that.
/// </para>
/// <para>
/// Tarjan's algorithm, written iteratively so a long chain cannot overflow the stack.
/// It completes a component only after every component reachable from it, which is
/// exactly the order heights have to be computed in, so they are taken in the same pass.
/// </para>
/// </summary>
public static class Condensation
{
    /// <summary>Folds the cycles of a directed graph and ranks what is left.</summary>
    /// <param name="nodeCount">Number of nodes. A node on no arc is a component by itself, at height zero.</param>
    /// <param name="arcs">Tail and head of each arc. Repeats and self-loops are ignored.</param>
    public static CondensationResult Of(int nodeCount, IEnumerable<(int From, int To)> arcs)
    {
        ArgumentNullException.ThrowIfNull(arcs);
        if (nodeCount < 1)
            throw new ArgumentOutOfRangeException(nameof(nodeCount), nodeCount, "Need at least one node.");

        var heads = new SortedSet<int>[nodeCount];
        foreach (var (from, to) in arcs)
        {
            if (from < 0 || from >= nodeCount || to < 0 || to >= nodeCount)
                throw new ArgumentOutOfRangeException(nameof(arcs), $"Arc ({from}, {to}) refers to a node outside 0..{nodeCount - 1}.");
            if (from != to)
                (heads[from] ??= new SortedSet<int>()).Add(to);
        }

        var next = heads.Select(set => set?.ToArray() ?? Array.Empty<int>()).ToArray();

        var index = new int[nodeCount];
        var low = new int[nodeCount];
        var cursor = new int[nodeCount];
        var component = new int[nodeCount];
        var open = new bool[nodeCount];
        Array.Fill(index, -1);
        Array.Fill(component, -1);

        var heights = new List<int>();
        var pending = new Stack<int>();
        var calls = new Stack<int>();
        int clock = 0;

        for (int root = 0; root < nodeCount; root++)
        {
            if (index[root] >= 0)
                continue;

            calls.Push(root);
            while (calls.Count > 0)
            {
                int v = calls.Peek();
                if (index[v] < 0)
                {
                    index[v] = low[v] = clock++;
                    pending.Push(v);
                    open[v] = true;
                }

                bool descended = false;
                while (cursor[v] < next[v].Length)
                {
                    int w = next[v][cursor[v]++];
                    if (index[w] < 0)
                    {
                        calls.Push(w);
                        descended = true;
                        break;
                    }

                    if (open[w])
                        low[v] = Math.Min(low[v], index[w]);
                }

                if (descended)
                    continue;

                calls.Pop();
                if (calls.Count > 0)
                    low[calls.Peek()] = Math.Min(low[calls.Peek()], low[v]);

                if (low[v] != index[v])
                    continue;

                int c = heights.Count;
                var nodes = new List<int>();
                int w2;
                do
                {
                    w2 = pending.Pop();
                    open[w2] = false;
                    component[w2] = c;
                    nodes.Add(w2);
                }
                while (w2 != v);

                int height = 0;
                foreach (int node in nodes)
                    foreach (int head in next[node])
                        if (component[head] != c)
                            height = Math.Max(height, heights[component[head]] + 1);

                heights.Add(height);
            }
        }

        // Renumber by lowest node, so the numbering is the graph's and not the search's.
        var renumber = new int[heights.Count];
        Array.Fill(renumber, -1);
        int count = 0;
        for (int v = 0; v < nodeCount; v++)
            if (renumber[component[v]] < 0)
                renumber[component[v]] = count++;

        var orderedHeights = new int[count];
        for (int c = 0; c < count; c++)
            orderedHeights[renumber[c]] = heights[c];

        return new CondensationResult(component.Select(c => renumber[c]).ToArray(), count, orderedHeights);
    }
}
