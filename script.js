// Create animated background elements
function createAnimatedBackground() {
    const leafBg = document.createElement('div');
    leafBg.className = 'leaf-bg';
    document.body.appendChild(leafBg);
    
    const particles = document.createElement('div');
    particles.className = 'floating-particles';
    document.body.appendChild(particles);
    
    // Create floating particles
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.width = Math.random() * 10 + 5 + 'px';
        particle.style.height = particle.style.width;
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = Math.random() * 20 + 10 + 's';
        particle.style.background = `rgba(${Math.random() * 100 + 100}, ${Math.random() * 100 + 155}, ${Math.random() * 100 + 100}, 0.5)`;
        particles.appendChild(particle);
    }
}

// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const currentTheme = localStorage.getItem('theme') || 'day';

function applyTheme(theme) {
    const root = document.documentElement;
    
    if (theme === 'night') {
        root.setAttribute('data-theme', 'night');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i> Switch to Day Mode';
    } else {
        root.setAttribute('data-theme', 'day');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i> Switch to Night Mode';
    }
    
    localStorage.setItem('theme', theme);
}

applyTheme(currentTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'night' ? 'day' : 'night';
    applyTheme(newTheme);
});

// ========== TAB SWITCHING FUNCTION ==========
function initializeTabs() {
    // Get all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    // Add click event to each tab button
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Get the parent section
            const section = this.closest('section');
            if (!section) return;
            
            // Get the target tab content ID
            const targetId = this.getAttribute('data-target');
            
            // Remove active class from all tab buttons in this section
            const allButtons = section.querySelectorAll('.tab-btn');
            allButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Hide all tab contents in this section
            const allContents = section.querySelectorAll('.tab-content');
            allContents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });
            
            // Show the target tab content
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
                targetContent.style.display = 'block';
            }
        });
    });
}

// Algorithm Code Data
const algorithmCode = {
    binarySearch: {
        cpp: `#include <iostream>
#include <vector>
using namespace std;

int binarySearch(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int target = 13;
    
    int result = binarySearch(arr, target);
    if (result != -1) {
        cout << "Element found at index: " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }
    
    return 0;
}`,
        python: `def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 13

result = binary_search(arr, target)
if result != -1:
    print(f"Element found at index: {result}")
else:
    print("Element not found")`
    },
    quickSort: {
        cpp: `#include <iostream>
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 5, 3, 2, 4, 6};
    int n = arr.size();
    
    cout << "Original array: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    
    quickSort(arr, 0, n - 1);
    
    cout << "Sorted array: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    
    return 0;
}`,
        python: `def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

# Example usage
arr = [10, 7, 8, 9, 1, 5, 3, 2, 4, 6]
n = len(arr)

print("Original array:", arr)
quick_sort(arr, 0, n - 1)
print("Sorted array:", arr)`
    },
    BFS: {
        cpp: `#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
using namespace std;

vector<int> bfs(const vector<vector<int>>& graph, int start) {
    vector<int> result;
    unordered_set<int> visited;
    queue<int> q;
    
    q.push(start);
    visited.insert(start);
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);
        
        for (int neighbor : graph[current]) {
            if (visited.find(neighbor) == visited.end()) {
                q.push(neighbor);
                visited.insert(neighbor);
            }
        }
    }
    
    return result;
}

int main() {
    // Example graph: adjacency list
    vector<vector<int>> graph = {
        {1, 2},    // Node 0
        {0, 3, 4}, // Node 1
        {0, 5},    // Node 2
        {1},       // Node 3
        {1, 6},    // Node 4
        {2},       // Node 5
        {4}        // Node 6
    };
    
    vector<int> traversal = bfs(graph, 0);
    
    cout << "BFS traversal starting from node 0: ";
    for (int node : traversal) {
        cout << node << " ";
    }
    cout << endl;
    
    return 0;
}`,
        python: `from collections import deque

def bfs(graph, start):
    result = []
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    
    return result

# Example usage
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1],
    4: [1, 6],
    5: [2],
    6: [4]
}

traversal = bfs(graph, 0)
print("BFS traversal starting from node 0:", traversal)`
    },
    DFS: {
        cpp: `#include <iostream>
#include <vector>
#include <stack>
#include <unordered_set>
using namespace std;

vector<int> dfs(const vector<vector<int>>& graph, int start) {
    vector<int> result;
    unordered_set<int> visited;
    stack<int> s;
    
    s.push(start);
    
    while (!s.empty()) {
        int current = s.top();
        s.pop();
        
        if (visited.find(current) == visited.end()) {
            visited.insert(current);
            result.push_back(current);
            
            // Push neighbors in reverse order for proper traversal
            for (auto it = graph[current].rbegin(); it != graph[current].rend(); ++it) {
                s.push(*it);
            }
        }
    }
    
    return result;
}

int main() {
    vector<vector<int>> graph = {
        {1, 2},
        {0, 3, 4},
        {0, 5},
        {1},
        {1, 6},
        {2},
        {4}
    };
    
    vector<int> traversal = dfs(graph, 0);
    
    cout << "DFS traversal starting from node 0: ";
    for (int node : traversal) {
        cout << node << " ";
    }
    cout << endl;
    
    return 0;
}`,
        python: `def dfs(graph, start):
    result = []
    visited = set()
    stack = [start]
    
    while stack:
        current = stack.pop()
        
        if current not in visited:
            visited.add(current)
            result.append(current)
            
            # Add neighbors in reverse order for proper traversal
            for neighbor in reversed(graph[current]):
                stack.append(neighbor)
    
    return result

# Example usage
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1],
    4: [1, 6],
    5: [2],
    6: [4]
}

traversal = dfs(graph, 0)
print("DFS traversal starting from node 0:", traversal)`
    },
    mergeSort: {
        cpp: `#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    vector<int> leftArr(n1), rightArr(n2);
    
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        
        merge(arr, left, mid, right);
    }
}

int main() {
    vector<int> arr = {38, 27, 43, 3, 9, 82, 10, 15, 21, 7};
    int n = arr.size();
    
    cout << "Original array: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    
    mergeSort(arr, 0, n - 1);
    
    cout << "Sorted array: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    
    return 0;
}`,
        python: `def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        merge_sort(left_half)
        merge_sort(right_half)
        
        i = j = k = 0
        
        while i < len(left_half) and j < len(right_half):
            if left_half[i] <= right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10, 15, 21, 7]
print("Original array:", arr)

merge_sort(arr)
print("Sorted array:", arr)`
    },
    dijkstra: {
        cpp: `#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

vector<int> dijkstra(const vector<vector<pair<int, int>>>& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[source] = 0;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int currentDist = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (currentDist > dist[u]) continue;
        
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

int main() {
    // Graph: adjacency list with (node, weight) pairs
    int n = 5;
    vector<vector<pair<int, int>>> graph(n);
    
    // Add edges
    graph[0].push_back({1, 4});
    graph[0].push_back({2, 1});
    graph[1].push_back({3, 1});
    graph[2].push_back({1, 2});
    graph[2].push_back({3, 5});
    graph[3].push_back({4, 3});
    
    int source = 0;
    vector<int> distances = dijkstra(graph, source);
    
    cout << "Shortest distances from node " << source << ":" << endl;
    for (int i = 0; i < n; i++) {
        cout << "Node " << i << ": ";
        if (distances[i] == INT_MAX)
            cout << "INF" << endl;
        else
            cout << distances[i] << endl;
    }
    
    return 0;
}`,
        python: `import heapq

def dijkstra(graph, source):
    n = len(graph)
    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return dist

# Example usage
graph = [
    [(1, 4), (2, 1)],  # Node 0
    [(3, 1)],           # Node 1
    [(1, 2), (3, 5)],   # Node 2
    [(4, 3)],           # Node 3
    []                  # Node 4
]

source = 0
distances = dijkstra(graph, source)

print(f"Shortest distances from node {source}:")
for i in range(len(distances)):
    print(f"Node {i}: {distances[i] if distances[i] != float('inf') else 'INF'}")`
    },
    kadane: {
        cpp: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int kadane(const vector<int>& arr) {
    int max_current = arr[0];
    int max_global = arr[0];
    
    for (int i = 1; i < arr.size(); i++) {
        max_current = max(arr[i], max_current + arr[i]);
        if (max_current > max_global) {
            max_global = max_current;
        }
    }
    
    return max_global;
}

int main() {
    vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    
    cout << "Array: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    
    int max_sum = kadane(arr);
    cout << "Maximum subarray sum: " << max_sum << endl;
    
    // Extended version to find subarray indices
    int start = 0, end = 0, temp_start = 0;
    max_current = arr[0];
    max_global = arr[0];
    
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > max_current + arr[i]) {
            max_current = arr[i];
            temp_start = i;
        } else {
            max_current = max_current + arr[i];
        }
        
        if (max_current > max_global) {
            max_global = max_current;
            start = temp_start;
            end = i;
        }
    }
    
    cout << "Maximum sum subarray: ";
    for (int i = start; i <= end; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    return 0;
}`,
        python: `def kadane(arr):
    max_current = max_global = arr[0]
    
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        if max_current > max_global:
            max_global = max_current
    
    return max_global

# Extended version to find subarray indices
def kadane_with_indices(arr):
    max_current = max_global = arr[0]
    start = end = temp_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_current + arr[i]:
            max_current = arr[i]
            temp_start = i
        else:
            max_current = max_current + arr[i]
        
        if max_current > max_global:
            max_global = max_current
            start = temp_start
            end = i
    
    return max_global, start, end

# Example usage
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print("Array:", arr)

max_sum = kadane(arr)
print("Maximum subarray sum:", max_sum)

max_sum, start, end = kadane_with_indices(arr)
print(f"Maximum sum subarray (indices {start}-{end}):", arr[start:end+1])`
    },
    knapsack: {
        cpp: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int knapsack(int capacity, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], 
                              dp[i - 1][w]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    
    return dp[n][capacity];
}

int main() {
    vector<int> values = {60, 100, 120};
    vector<int> weights = {10, 20, 30};
    int capacity = 50;
    
    cout << "Values: ";
    for (int v : values) cout << v << " ";
    cout << endl;
    
    cout << "Weights: ";
    for (int w : weights) cout << w << " ";
    cout << endl;
    
    cout << "Capacity: " << capacity << endl;
    
    int max_value = knapsack(capacity, weights, values);
    cout << "Maximum value in knapsack: " << max_value << endl;
    
    return 0;
}`,
        python: `def knapsack(capacity, weights, values):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], 
                              dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

# Optimized space version
def knapsack_optimized(capacity, weights, values):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    
    return dp[capacity]

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

print("Values:", values)
print("Weights:", weights)
print("Capacity:", capacity)

max_value = knapsack(capacity, weights, values)
print("Maximum value in knapsack:", max_value)

max_value_opt = knapsack_optimized(capacity, weights, values)
print("Maximum value (optimized):", max_value_opt)`
    },
    floydWarshall: {
        cpp: `#include <iostream>
#include <vector>
#include <climits>
using namespace std;

const int INF = INT_MAX / 2;

void floydWarshall(vector<vector<int>>& dist) {
    int V = dist.size();
    
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

void printSolution(const vector<vector<int>>& dist) {
    int V = dist.size();
    cout << "Shortest distances between every pair of vertices:" << endl;
    
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                cout << "INF\t";
            else
                cout << dist[i][j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    int V = 4;
    vector<vector<int>> graph = {
        {0, 5, INF, 10},
        {INF, 0, 3, INF},
        {INF, INF, 0, 1},
        {INF, INF, INF, 0}
    };
    
    // Make a copy for Floyd-Warshall
    vector<vector<int>> dist = graph;
    
    floydWarshall(dist);
    printSolution(dist);
    
    return 0;
}`,
        python: `def floyd_warshall(graph):
    V = len(graph)
    dist = [row[:] for row in graph]
    
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf') and \
                   dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def print_solution(dist):
    V = len(dist)
    print("Shortest distances between every pair of vertices:")
    
    for i in range(V):
        for j in range(V):
            if dist[i][j] == float('inf'):
                print("INF", end="\t")
            else:
                print(dist[i][j], end="\t")
        print()

# Example usage
V = 4
INF = float('inf')
graph = [
    [0, 5, INF, 10],
    [INF, 0, 3, INF],
    [INF, INF, 0, 1],
    [INF, INF, INF, 0]
]

distances = floyd_warshall(graph)
print_solution(distances)`
    },
    fibonacci: {
        cpp: `#include <iostream>
#include <vector>
using namespace std;

// Recursive Fibonacci (exponential time)
int fibRecursive(int n) {
    if (n <= 1) return n;
    return fibRecursive(n - 1) + fibRecursive(n - 2);
}

// DP Fibonacci (O(n) time, O(n) space)
int fibDP(int n) {
    if (n <= 1) return n;
    
    vector<int> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}

// Optimized DP Fibonacci (O(n) time, O(1) space)
int fibOptimized(int n) {
    if (n <= 1) return n;
    
    int prev2 = 0;  // F(n-2)
    int prev1 = 1;  // F(n-1)
    int current;
    
    for (int i = 2; i <= n; i++) {
        current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return current;
}

// Matrix Exponentiation Fibonacci (O(log n) time)
void multiply(int F[2][2], int M[2][2]) {
    int x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
    int y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
    int z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
    int w = F[1][0] * M[0][1] + F[1][1] * M[1][1];
    
    F[0][0] = x;
    F[0][1] = y;
    F[1][0] = z;
    F[1][1] = w;
}

void power(int F[2][2], int n) {
    if (n <= 1) return;
    
    int M[2][2] = {{1, 1}, {1, 0}};
    
    power(F, n / 2);
    multiply(F, F);
    
    if (n % 2 != 0) {
        multiply(F, M);
    }
}

int fibMatrix(int n) {
    if (n <= 1) return n;
    
    int F[2][2] = {{1, 1}, {1, 0}};
    power(F, n - 1);
    
    return F[0][0];
}

int main() {
    int n = 10;
    
    cout << "Fibonacci numbers up to " << n << ":" << endl;
    
    cout << "Recursive: ";
    for (int i = 0; i <= n; i++) {
        cout << fibRecursive(i) << " ";
    }
    cout << endl;
    
    cout << "DP: ";
    for (int i = 0; i <= n; i++) {
        cout << fibDP(i) << " ";
    }
    cout << endl;
    
    cout << "Optimized DP: ";
    for (int i = 0; i <= n; i++) {
        cout << fibOptimized(i) << " ";
    }
    cout << endl;
    
    cout << "Matrix: ";
    for (int i = 0; i <= n; i++) {
        cout << fibMatrix(i) << " ";
    }
    cout << endl;
    
    return 0;
}`,
        python: `# Recursive Fibonacci (exponential time)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# DP Fibonacci (O(n) time, O(n) space)
def fib_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Optimized DP Fibonacci (O(n) time, O(1) space)
def fib_optimized(n):
    if n <= 1:
        return n
    
    prev2 = 0  # F(n-2)
    prev1 = 1  # F(n-1)
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return current

# Matrix Exponentiation Fibonacci (O(log n) time)
def multiply(F, M):
    x = F[0][0] * M[0][0] + F[0][1] * M[1][0]
    y = F[0][0] * M[0][1] + F[0][1] * M[1][1]
    z = F[1][0] * M[0][0] + F[1][1] * M[1][0]
    w = F[1][0] * M[0][1] + F[1][1] * M[1][1]
    
    F[0][0] = x
    F[0][1] = y
    F[1][0] = z
    F[1][1] = w

def power(F, n):
    if n <= 1:
        return
    
    M = [[1, 1], [1, 0]]
    power(F, n // 2)
    multiply(F, F)
    
    if n % 2 != 0:
        multiply(F, M)

def fib_matrix(n):
    if n <= 1:
        return n
    
    F = [[1, 1], [1, 0]]
    power(F, n - 1)
    return F[0][0]

# Example usage
n = 10
print(f"Fibonacci numbers up to {n}:")
print("Recursive:", [fib_recursive(i) for i in range(n + 1)])
print("DP:", [fib_dp(i) for i in range(n + 1)])
print("Optimized DP:", [fib_optimized(i) for i in range(n + 1)])
print("Matrix:", [fib_matrix(i) for i in range(n + 1)])`
    }
};

// Modal Functionality
const modal = document.getElementById('codeModal');
const modalTitle = document.getElementById('modalTitle');
const modalCppCode = document.getElementById('modalCppCode');
const modalPyCode = document.getElementById('modalPyCode');
let currentAlgorithm = '';

// Close modal when clicking outside
window.addEventListener('click', (event) => {
    if (event.target === modal) {
        closeModal();
    }
});

function showCode(algorithm) {
    currentAlgorithm = algorithm;
    modal.style.display = 'flex';
    
    // Set modal title based on algorithm
    const algoNames = {
        'binarySearch': 'Binary Search',
        'quickSort': 'Quick Sort',
        'BFS': 'Breadth-First Search',
        'DFS': 'Depth-First Search',
        'mergeSort': 'Merge Sort',
        'dijkstra': 'Dijkstra\'s Algorithm',
        'kadane': 'Kadane\'s Algorithm',
        'knapsack': '0/1 Knapsack',
        'floydWarshall': 'Floyd-Warshall',
        'fibonacci': 'Fibonacci Sequence'
    };
    
    modalTitle.textContent = `${algoNames[algorithm]} Implementation`;
    
    // Set code content
    if (algorithmCode[algorithm]) {
        modalCppCode.textContent = algorithmCode[algorithm].cpp;
        modalPyCode.textContent = algorithmCode[algorithm].python;
    }
}

function closeModal() {
    modal.style.display = 'none';
}

function switchModalTab(tab) {
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.modal-code').forEach(c => c.classList.remove('active'));
    
    document.querySelector(`.modal-tab[onclick*="${tab}"]`).classList.add('active');
    document.getElementById(`modal${tab.charAt(0).toUpperCase() + tab.slice(1)}`).classList.add('active');
}

function downloadModalCode() {
    const currentTab = document.querySelector('.modal-tab.active');
    if (!currentTab) return;
    
    const language = currentTab.textContent.toLowerCase();
    const algoNames = {
        'binarySearch': 'binary_search',
        'quickSort': 'quick_sort',
        'BFS': 'bfs',
        'DFS': 'dfs',
        'mergeSort': 'merge_sort',
        'dijkstra': 'dijkstra',
        'kadane': 'kadane',
        'knapsack': 'knapsack',
        'floydWarshall': 'floyd_warshall',
        'fibonacci': 'fibonacci'
    };
    
    const filename = `${algoNames[currentAlgorithm]}.${language === 'c++' ? 'cpp' : 'py'}`;
    const code = language === 'c++' ? algorithmCode[currentAlgorithm].cpp : algorithmCode[currentAlgorithm].python;
    
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification(`Downloaded: ${filename}`);
}

// Download Function
function downloadCode(elementId, filename) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const code = element.textContent;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification(`Downloaded: ${filename}`);
}

function downloadAllAlgorithms() {
    showNotification('Download feature coming soon!');
}

// Notification System
function showNotification(message) {
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 1000;
        opacity: 0;
        transform: translateX(100px);
        transition: all 0.3s ease;
        font-weight: 600;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100px)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Smooth Scrolling
function jump(id) {
    document.getElementById(id).scrollIntoView({
        behavior: 'smooth'
    });
}

// ========== ARRAY VISUALIZATION ==========
let array = [20, 12, 25, 8, 15, 30, 10, 18];
let sorting = false;

function renderArray() {
    const container = document.getElementById('arrayVisualization');
    if (!container) return;
    
    container.innerHTML = '';
    
    const maxValue = Math.max(...array);
    
    array.forEach((value, index) => {
        const bar = document.createElement('div');
        bar.className = 'array-bar';
        bar.textContent = value;
        bar.style.height = `${(value / maxValue) * 180}px`;
        container.appendChild(bar);
    });
}

function startBubbleSort() {
    if (sorting) return;
    sorting = true;
    
    let i = 0;
    let j = 0;
    const n = array.length;
    
    function bubbleStep() {
        if (i < n - 1) {
            if (j < n - i - 1) {
                if (array[j] > array[j + 1]) {
                    [array[j], array[j + 1]] = [array[j + 1], array[j]];
                    renderArray();
                }
                j++;
                setTimeout(bubbleStep, 300);
            } else {
                j = 0;
                i++;
                setTimeout(bubbleStep, 200);
            }
        } else {
            sorting = false;
            showNotification('Array sorted!');
        }
    }
    
    bubbleStep();
}

function resetArray() {
    array = [20, 12, 25, 8, 15, 30, 10, 18];
    renderArray();
}

function generateRandomArray() {
    array = Array.from({length: 8}, () => Math.floor(Math.random() * 50) + 1);
    renderArray();
}

// ========== STACK VISUALIZATION ==========
let stack = [];

function renderStack() {
    const container = document.getElementById('stackVisualization');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (stack.length === 0) {
        container.innerHTML = '<div class="stack-empty">Stack is empty</div>';
        return;
    }
    
    stack.slice().reverse().forEach(item => {
        const element = document.createElement('div');
        element.className = 'stack-item';
        element.textContent = item;
        container.appendChild(element);
    });
}

function pushStack() {
    const input = document.getElementById('stackInput');
    const value = input.value.trim();
    
    if (value) {
        stack.push(value);
        input.value = '';
        renderStack();
    }
}

function popStack() {
    if (stack.length > 0) {
        stack.pop();
        renderStack();
    }
}

function clearStack() {
    stack = [];
    renderStack();
}

// ========== QUEUE VISUALIZATION ==========
let queue = [];

function renderQueue() {
    const container = document.getElementById('queueVisualization');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (queue.length === 0) {
        container.innerHTML = '<div class="queue-empty">Queue is empty</div>';
        return;
    }
    
    queue.forEach(item => {
        const element = document.createElement('div');
        element.className = 'queue-item';
        element.textContent = item;
        container.appendChild(element);
    });
}

function enqueueQueue() {
    const input = document.getElementById('queueInput');
    const value = input.value.trim();
    
    if (value) {
        queue.push(value);
        input.value = '';
        renderQueue();
    }
}

function dequeueQueue() {
    if (queue.length > 0) {
        queue.shift();
        renderQueue();
    }
}

function clearQueue() {
    queue = [];
    renderQueue();
}

// ========== TREE VISUALIZATION ==========
const tree = {
    value: '10',
    left: {
        value: '5',
        left: { value: '3', left: null, right: null },
        right: { value: '7', left: null, right: null }
    },
    right: {
        value: '15',
        left: { value: '12', left: null, right: null },
        right: { value: '18', left: null, right: null }
    }
};

let isAnimating = false;
let animationSpeed = 1000;

function renderTree() {
    const container = document.getElementById('treeVisualization');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Create main container
    const treeDiagram = document.createElement('div');
    treeDiagram.style.cssText = `
        position: relative;
        width: 100%;
        height: 350px;
        display: flex;
        flex-direction: column;
        align-items: center;
    `;
    
    // Create SVG for connections
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        z-index: 1;
    `;
    
    // Level 1: Root node (10)
    const rootNode = createTreeNode(10, 400, 50, 'level-1');
    
    // Level 2: Children nodes (5, 15)
    const leftNode = createTreeNode(5, 250, 150, 'level-2');
    const rightNode = createTreeNode(15, 550, 150, 'level-2');
    
    // Level 3: Grandchildren (3, 7, 12, 18)
    const node3 = createTreeNode(3, 175, 250, 'level-3');
    const node7 = createTreeNode(7, 325, 250, 'level-3');
    const node12 = createTreeNode(12, 475, 250, 'level-3');
    const node18 = createTreeNode(18, 625, 250, 'level-3');
    
    // Draw connections
    drawConnection(svg, 400, 90, 250, 150);  // 10 -> 5
    drawConnection(svg, 400, 90, 550, 150);  // 10 -> 15
    drawConnection(svg, 250, 190, 175, 250); // 5 -> 3
    drawConnection(svg, 250, 190, 325, 250); // 5 -> 7
    drawConnection(svg, 550, 190, 475, 250); // 15 -> 12
    drawConnection(svg, 550, 190, 625, 250); // 15 -> 18
    
    // Add all elements to container
    treeDiagram.appendChild(svg);
    treeDiagram.appendChild(rootNode);
    treeDiagram.appendChild(leftNode);
    treeDiagram.appendChild(rightNode);
    treeDiagram.appendChild(node3);
    treeDiagram.appendChild(node7);
    treeDiagram.appendChild(node12);
    treeDiagram.appendChild(node18);
    
   
    container.appendChild(treeDiagram);
    
    // Store node references for animation
    window.treeNodeElements = {
        '10': rootNode,
        '5': leftNode,
        '15': rightNode,
        '3': node3,
        '7': node7,
        '12': node12,
        '18': node18
    };
}

function createTreeNode(value, x, y, levelClass) {
    const node = document.createElement('div');
    node.className = `tree-node ${levelClass}`;
    node.setAttribute('data-value', value);
    node.textContent = value;
    node.style.cssText = `
        position: absolute;
        width: 60px;
        height: 60px;
        left: ${x - 30}px;
        top: ${y}px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px var(--shadow);
        transition: all 0.3s ease;
        border: 3px solid transparent;
        z-index: 2;
        cursor: pointer;
    `;
    
    // Add hover effect
    node.addEventListener('mouseenter', () => {
        if (!node.classList.contains('highlight')) {
            node.style.transform = 'scale(1.1)';
            node.style.boxShadow = '0 6px 20px var(--shadow)';
        }
    });
    
    node.addEventListener('mouseleave', () => {
        if (!node.classList.contains('highlight')) {
            node.style.transform = 'scale(1)';
            node.style.boxShadow = '0 4px 12px var(--shadow)';
        }
    });
    
    return node;
}

function drawConnection(svg, x1, y1, x2, y2) {
    const svgNS = "http://www.w3.org/2000/svg";
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke", "var(--accent)");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
}

function traverseTree(type) {
    if (isAnimating) {
        showNotification('Animation in progress. Please wait.');
        return;
    }
    
    isAnimating = true;
    const output = document.getElementById('traversalOutput');
    output.innerHTML = `<span style="color: var(--accent)">${type.toUpperCase()} Traversal:</span> Starting...`;
    
    const traversalOrder = getTraversalOrder(type);
    let step = 0;
    
    function animateStep() {
        if (step >= traversalOrder.length) {
            // Animation complete
            isAnimating = false;
            output.innerHTML = `<span style="color: var(--accent)">${type.toUpperCase()} Traversal:</span> ` +
                traversalOrder.map(val => `<span style="color: var(--accent-light)">${val}</span>`).join(' → ') +
                `<br><span style="color: gold; margin-top: 10px; display: block;">✓ Traversal Complete!</span>`;
            
            // Remove all highlights after completion
            setTimeout(() => {
                traversalOrder.forEach(value => {
                    const node = window.treeNodeElements[value];
                    if (node) {
                        node.classList.remove('highlight');
                        node.style.transform = 'scale(1)';
                        node.style.boxShadow = '0 4px 12px var(--shadow)';
                        node.style.border = '3px solid transparent';
                    }
                });
            }, 1000);
            
            return;
        }
        
        const value = traversalOrder[step];
        const node = window.treeNodeElements[value];
        
        if (node) {
            // Highlight current node
            node.classList.add('highlight');
            node.style.transform = 'scale(1.3)';
            node.style.boxShadow = '0 8px 25px var(--shadow)';
            node.style.border = '3px solid gold';
            node.style.zIndex = '10';
            
            // Update output
            const currentPath = traversalOrder.slice(0, step + 1);
            output.innerHTML = `<span style="color: var(--accent)">${type.toUpperCase()} Traversal:</span> ` +
                currentPath.map((val, idx) => 
                    `<span style="color: ${idx === currentPath.length - 1 ? 'gold' : 'var(--accent-light)'}">${val}</span>`
                ).join(' → ');
            
            // Remove highlight after delay, but keep the last one longer
            const removeDelay = (step === traversalOrder.length - 1) ? animationSpeed * 2 : animationSpeed / 2;
            
            setTimeout(() => {
                if (step !== traversalOrder.length - 1) {
                    node.classList.remove('highlight');
                    node.style.transform = 'scale(1.1)';
                    node.style.boxShadow = '0 6px 20px var(--shadow)';
                    node.style.border = '3px solid var(--accent-light)';
                    node.style.zIndex = '2';
                }
            }, removeDelay);
        }
        
        step++;
        setTimeout(animateStep, animationSpeed);
    }
    
    animateStep();
}

function getTraversalOrder(type) {
    const result = [];
    
    function inorder(node) {
        if (!node) return;
        inorder(node.left);
        result.push(node.value);
        inorder(node.right);
    }
    
    function preorder(node) {
        if (!node) return;
        result.push(node.value);
        preorder(node.left);
        preorder(node.right);
    }
    
    function postorder(node) {
        if (!node) return;
        postorder(node.left);
        postorder(node.right);
        result.push(node.value);
    }
    
    switch(type) {
        case 'inorder':
            inorder(tree);
            break;
        case 'preorder':
            preorder(tree);
            break;
        case 'postorder':
            postorder(tree);
            break;
    }
    
    return result;
}

function resetTree() {
    if (isAnimating) {
        showNotification('Please wait for animation to complete.');
        return;
    }
    
    const output = document.getElementById('traversalOutput');
    output.innerHTML = 'Click a traversal button to see the order';
    
    // Reset all nodes
    if (window.treeNodeElements) {
        Object.values(window.treeNodeElements).forEach(node => {
            if (node) {
                node.classList.remove('highlight');
                node.style.transform = 'scale(1)';
                node.style.boxShadow = '0 4px 12px var(--shadow)';
                node.style.border = '3px solid transparent';
                node.style.zIndex = '2';
            }
        });
    }
    
    showNotification('Tree visualization reset');
}

// Add CSS for animations
const treeStyles = document.createElement('style');
treeStyles.textContent = `
    .tree-node.highlight {
        animation: nodePulse 0.5s ease-in-out infinite alternate;
    }
    
    @keyframes nodePulse {
        0% {
            transform: scale(1.2);
            box-shadow: 0 0 20px gold, 0 0 40px rgba(255, 215, 0, 0.3);
        }
        100% {
            transform: scale(1.4);
            box-shadow: 0 0 30px gold, 0 0 60px rgba(255, 215, 0, 0.5);
        }
    }
    
    .traversal-result {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--border);
        box-shadow: 0 4px 20px var(--shadow);
    }
    
    .traversal-result h4 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    #traversalOutput {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 1.1rem;
        line-height: 1.6;
        color: var(--text-primary);
        background: var(--code-bg);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        min-height: 120px;
        transition: all 0.3s ease;
    }
    
    .tree-vis {
        min-height: 450px;
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid var(--border);
        position: relative;
        overflow: hidden;
    }
    
    .tree-vis::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(46, 125, 50, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(76, 175, 80, 0.1) 0%, transparent 50%);
        z-index: 0;
    }
`;
document.head.appendChild(treeStyles);

// ========== INITIALIZATION ==========
document.addEventListener('DOMContentLoaded', () => {
    createAnimatedBackground();
    
    // Initialize tabs
    initializeTabs();
    
    // Set first tab as active in each section
    document.querySelectorAll('.tabs').forEach(tabs => {
        const firstBtn = tabs.querySelector('.tab-btn');
        if (firstBtn && !firstBtn.classList.contains('active')) {
            firstBtn.classList.add('active');
            const targetId = firstBtn.getAttribute('data-target');
            const target = document.getElementById(targetId);
            if (target) {
                target.classList.add('active');
                target.style.display = 'block';
            }
        }
    });
    
    // Hide all non-active tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        if (!content.classList.contains('active')) {
            content.style.display = 'none';
        }
    });
    
    // Initialize visualizations
    renderArray();
    renderStack();
    renderQueue();
    renderTree();
});

// Make functions available globally for HTML onclick events
window.startBubbleSort = startBubbleSort;
window.resetArray = resetArray;
window.generateRandomArray = generateRandomArray;
window.pushStack = pushStack;
window.popStack = popStack;
window.clearStack = clearStack;
window.enqueueQueue = enqueueQueue;
window.dequeueQueue = dequeueQueue;
window.clearQueue = clearQueue;
window.traverseTree = traverseTree;
window.resetTree = resetTree;
window.showCode = showCode;
window.closeModal = closeModal;
window.switchModalTab = switchModalTab;
window.downloadModalCode = downloadModalCode;
window.downloadCode = downloadCode;
window.downloadAllAlgorithms = downloadAllAlgorithms;
window.jump = jump;

// Mobile Navigation Toggle
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const navMenu = document.querySelector("header nav");

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
        navMenu.classList.toggle("show");
    });
}
// Slide-out sidebar drawer
const drawerBtn = document.getElementById("drawerBtn");
const sidebar = document.querySelector("aside.sidebar");

drawerBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});

// Close drawer on section click (mobile)
document.addEventListener("click", function(e) {
    if (window.innerWidth <= 900) {
        if (!sidebar.contains(e.target) && !drawerBtn.contains(e.target)) {
            sidebar.classList.remove("open");
        }
    }
});
// Swipeable algorithm cards (horizontal scroll)
const grids = document.querySelectorAll('.algorithm-grid');

grids.forEach(grid => {
    grid.style.overflowX = "auto";
    grid.style.scrollBehavior = "smooth";

    let startX = 0;
    let scrollLeft = 0;

    grid.addEventListener('touchstart', (e) => {
        startX = e.touches[0].pageX;
        scrollLeft = grid.scrollLeft;
    });

    grid.addEventListener('touchmove', (e) => {
        let x = e.touches[0].pageX;
        let walk = startX - x;
        grid.scrollLeft = scrollLeft + walk;
    });
});
// Touch gesture for sidebar swipe-open
let touchStartX = 0;

document.addEventListener("touchstart", (e) => {
    touchStartX = e.touches[0].clientX;
});

// Swipe right to open sidebar
document.addEventListener("touchmove", (e) => {
    let current = e.touches[0].clientX;

    if (touchStartX < 40 && current - touchStartX > 80) {
        sidebar.classList.add("open");
    }
});

// Swipe left to close sidebar
sidebar.addEventListener("touchmove", (e) => {
    let current = e.touches[0].clientX;

    if (touchStartX - current > 80) {
        sidebar.classList.remove("open");
    }
});
// Scroll-to-top button
const scrollBtn = document.getElementById("scrollTopBtn");

window.addEventListener("scroll", () => {
    if (window.scrollY > 400) {
        scrollBtn.classList.add("show");
    } else {
        scrollBtn.classList.remove("show");
    }
});

scrollBtn.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
});
