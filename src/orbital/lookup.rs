pub const VERTEX_RELATIVE_POSITION: [[f64; 4]; 8] = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
];

// Pair of vertex indices for each edge on the cube
pub const EDGE_VERTEX_INDICES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    [0, 4],
    [1, 5],
    [3, 7],
    [2, 6],
];

// For each MC case, a mask of edge indices that need to be split
pub const _EDGE_MASKS: [i32; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x80c, 0x905, 0xa0f, 0xb06, 0x406, 0x50f, 0x605, 0x70c, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x99c, 0x895, 0xb9f, 0xa96, 0x596, 0x49f, 0x795,
    0x69c, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0xa3c, 0xb35, 0x83f, 0x936,
    0x636, 0x73f, 0x435, 0x53c, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0xbac,
    0xaa5, 0x9af, 0x8a6, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xfaa, 0xea3, 0xda9, 0xca0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0x15c, 0x55, 0x35f, 0x256, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0x2fc, 0x3f5, 0xff, 0x1f6, 0xef6,
    0xfff, 0xcf5, 0xdfc, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0x36c, 0x265,
    0x16f, 0x66, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x76a, 0x663, 0x569, 0x460, 0x460, 0x569, 0x663,
    0x76a, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x66, 0x16f, 0x265, 0x36c, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x1f6, 0xff, 0x3f5, 0x2fc, 0x9fa,
    0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0xe5c, 0xf55, 0xc5f, 0xd56, 0x256, 0x35f,
    0x55, 0x15c, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0xfcc, 0xec5, 0xdcf,
    0xcc6, 0x3c6, 0x2cf, 0x1c5, 0xcc, 0xbca, 0xac3, 0x9c9, 0x8c0, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x8a6, 0x9af, 0xaa5, 0xbac, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x53c, 0x435, 0x73f, 0x636, 0x936, 0x83f, 0xb35, 0xa3c, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0x69c, 0x795, 0x49f, 0x596, 0xa96, 0xb9f, 0x895,
    0x99c, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0x70c, 0x605, 0x50f, 0x406,
    0xb06, 0xa0f, 0x905, 0x80c, 0x30a, 0x203, 0x109, 0x0,
];

// For each MC case, a list of triangles, specified as triples of edge indices, terminated by -1
pub fn triangle_table(n: usize) -> Vec<i8> {
    match n {
        0 => vec![-1],
        1 => vec![0, 3, 8, -1],
        2 => vec![0, 9, 1, -1],
        3 => vec![3, 8, 1, 1, 8, 9, -1],
        4 => vec![2, 11, 3, -1],
        5 => vec![8, 0, 11, 11, 0, 2, -1],
        6 => vec![3, 2, 11, 1, 0, 9, -1],
        7 => vec![11, 1, 2, 11, 9, 1, 11, 8, 9, -1],
        8 => vec![1, 10, 2, -1],
        9 => vec![0, 3, 8, 2, 1, 10, -1],
        10 => vec![10, 2, 9, 9, 2, 0, -1],
        11 => vec![8, 2, 3, 8, 10, 2, 8, 9, 10, -1],
        12 => vec![11, 3, 10, 10, 3, 1, -1],
        13 => vec![10, 0, 1, 10, 8, 0, 10, 11, 8, -1],
        14 => vec![9, 3, 0, 9, 11, 3, 9, 10, 11, -1],
        15 => vec![8, 9, 11, 11, 9, 10, -1],
        16 => vec![4, 8, 7, -1],
        17 => vec![7, 4, 3, 3, 4, 0, -1],
        18 => vec![4, 8, 7, 0, 9, 1, -1],
        19 => vec![1, 4, 9, 1, 7, 4, 1, 3, 7, -1],
        20 => vec![8, 7, 4, 11, 3, 2, -1],
        21 => vec![4, 11, 7, 4, 2, 11, 4, 0, 2, -1],
        22 => vec![0, 9, 1, 8, 7, 4, 11, 3, 2, -1],
        23 => vec![7, 4, 11, 11, 4, 2, 2, 4, 9, 2, 9, 1, -1],
        24 => vec![4, 8, 7, 2, 1, 10, -1],
        25 => vec![7, 4, 3, 3, 4, 0, 10, 2, 1, -1],
        26 => vec![10, 2, 9, 9, 2, 0, 7, 4, 8, -1],
        27 => vec![10, 2, 3, 10, 3, 4, 3, 7, 4, 9, 10, 4, -1],
        28 => vec![1, 10, 3, 3, 10, 11, 4, 8, 7, -1],
        29 => vec![10, 11, 1, 11, 7, 4, 1, 11, 4, 1, 4, 0, -1],
        30 => vec![7, 4, 8, 9, 3, 0, 9, 11, 3, 9, 10, 11, -1],
        31 => vec![7, 4, 11, 4, 9, 11, 9, 10, 11, -1],
        32 => vec![9, 4, 5, -1],
        33 => vec![9, 4, 5, 8, 0, 3, -1],
        34 => vec![4, 5, 0, 0, 5, 1, -1],
        35 => vec![5, 8, 4, 5, 3, 8, 5, 1, 3, -1],
        36 => vec![9, 4, 5, 11, 3, 2, -1],
        37 => vec![2, 11, 0, 0, 11, 8, 5, 9, 4, -1],
        38 => vec![4, 5, 0, 0, 5, 1, 11, 3, 2, -1],
        39 => vec![5, 1, 4, 1, 2, 11, 4, 1, 11, 4, 11, 8, -1],
        40 => vec![1, 10, 2, 5, 9, 4, -1],
        41 => vec![9, 4, 5, 0, 3, 8, 2, 1, 10, -1],
        42 => vec![2, 5, 10, 2, 4, 5, 2, 0, 4, -1],
        43 => vec![10, 2, 5, 5, 2, 4, 4, 2, 3, 4, 3, 8, -1],
        44 => vec![11, 3, 10, 10, 3, 1, 4, 5, 9, -1],
        45 => vec![4, 5, 9, 10, 0, 1, 10, 8, 0, 10, 11, 8, -1],
        46 => vec![11, 3, 0, 11, 0, 5, 0, 4, 5, 10, 11, 5, -1],
        47 => vec![4, 5, 8, 5, 10, 8, 10, 11, 8, -1],
        48 => vec![8, 7, 9, 9, 7, 5, -1],
        49 => vec![3, 9, 0, 3, 5, 9, 3, 7, 5, -1],
        50 => vec![7, 0, 8, 7, 1, 0, 7, 5, 1, -1],
        51 => vec![7, 5, 3, 3, 5, 1, -1],
        52 => vec![5, 9, 7, 7, 9, 8, 2, 11, 3, -1],
        53 => vec![2, 11, 7, 2, 7, 9, 7, 5, 9, 0, 2, 9, -1],
        54 => vec![2, 11, 3, 7, 0, 8, 7, 1, 0, 7, 5, 1, -1],
        55 => vec![2, 11, 1, 11, 7, 1, 7, 5, 1, -1],
        56 => vec![8, 7, 9, 9, 7, 5, 2, 1, 10, -1],
        57 => vec![10, 2, 1, 3, 9, 0, 3, 5, 9, 3, 7, 5, -1],
        58 => vec![7, 5, 8, 5, 10, 2, 8, 5, 2, 8, 2, 0, -1],
        59 => vec![10, 2, 5, 2, 3, 5, 3, 7, 5, -1],
        60 => vec![8, 7, 5, 8, 5, 9, 11, 3, 10, 3, 1, 10, -1],
        61 => vec![5, 11, 7, 10, 11, 5, 1, 9, 0, -1],
        62 => vec![11, 5, 10, 7, 5, 11, 8, 3, 0, -1],
        63 => vec![5, 11, 7, 10, 11, 5, -1],
        64 => vec![6, 7, 11, -1],
        65 => vec![7, 11, 6, 3, 8, 0, -1],
        66 => vec![6, 7, 11, 0, 9, 1, -1],
        67 => vec![9, 1, 8, 8, 1, 3, 6, 7, 11, -1],
        68 => vec![3, 2, 7, 7, 2, 6, -1],
        69 => vec![0, 7, 8, 0, 6, 7, 0, 2, 6, -1],
        70 => vec![6, 7, 2, 2, 7, 3, 9, 1, 0, -1],
        71 => vec![6, 7, 8, 6, 8, 1, 8, 9, 1, 2, 6, 1, -1],
        72 => vec![11, 6, 7, 10, 2, 1, -1],
        73 => vec![3, 8, 0, 11, 6, 7, 10, 2, 1, -1],
        74 => vec![0, 9, 2, 2, 9, 10, 7, 11, 6, -1],
        75 => vec![6, 7, 11, 8, 2, 3, 8, 10, 2, 8, 9, 10, -1],
        76 => vec![7, 10, 6, 7, 1, 10, 7, 3, 1, -1],
        77 => vec![8, 0, 7, 7, 0, 6, 6, 0, 1, 6, 1, 10, -1],
        78 => vec![7, 3, 6, 3, 0, 9, 6, 3, 9, 6, 9, 10, -1],
        79 => vec![6, 7, 10, 7, 8, 10, 8, 9, 10, -1],
        80 => vec![11, 6, 8, 8, 6, 4, -1],
        81 => vec![6, 3, 11, 6, 0, 3, 6, 4, 0, -1],
        82 => vec![11, 6, 8, 8, 6, 4, 1, 0, 9, -1],
        83 => vec![1, 3, 9, 3, 11, 6, 9, 3, 6, 9, 6, 4, -1],
        84 => vec![2, 8, 3, 2, 4, 8, 2, 6, 4, -1],
        85 => vec![4, 0, 6, 6, 0, 2, -1],
        86 => vec![9, 1, 0, 2, 8, 3, 2, 4, 8, 2, 6, 4, -1],
        87 => vec![9, 1, 4, 1, 2, 4, 2, 6, 4, -1],
        88 => vec![4, 8, 6, 6, 8, 11, 1, 10, 2, -1],
        89 => vec![1, 10, 2, 6, 3, 11, 6, 0, 3, 6, 4, 0, -1],
        90 => vec![11, 6, 4, 11, 4, 8, 10, 2, 9, 2, 0, 9, -1],
        91 => vec![10, 4, 9, 6, 4, 10, 11, 2, 3, -1],
        92 => vec![4, 8, 3, 4, 3, 10, 3, 1, 10, 6, 4, 10, -1],
        93 => vec![1, 10, 0, 10, 6, 0, 6, 4, 0, -1],
        94 => vec![4, 10, 6, 9, 10, 4, 0, 8, 3, -1],
        95 => vec![4, 10, 6, 9, 10, 4, -1],
        96 => vec![6, 7, 11, 4, 5, 9, -1],
        97 => vec![4, 5, 9, 7, 11, 6, 3, 8, 0, -1],
        98 => vec![1, 0, 5, 5, 0, 4, 11, 6, 7, -1],
        99 => vec![11, 6, 7, 5, 8, 4, 5, 3, 8, 5, 1, 3, -1],
        100 => vec![3, 2, 7, 7, 2, 6, 9, 4, 5, -1],
        101 => vec![5, 9, 4, 0, 7, 8, 0, 6, 7, 0, 2, 6, -1],
        102 => vec![3, 2, 6, 3, 6, 7, 1, 0, 5, 0, 4, 5, -1],
        103 => vec![6, 1, 2, 5, 1, 6, 4, 7, 8, -1],
        104 => vec![10, 2, 1, 6, 7, 11, 4, 5, 9, -1],
        105 => vec![0, 3, 8, 4, 5, 9, 11, 6, 7, 10, 2, 1, -1],
        106 => vec![7, 11, 6, 2, 5, 10, 2, 4, 5, 2, 0, 4, -1],
        107 => vec![8, 4, 7, 5, 10, 6, 3, 11, 2, -1],
        108 => vec![9, 4, 5, 7, 10, 6, 7, 1, 10, 7, 3, 1, -1],
        109 => vec![10, 6, 5, 7, 8, 4, 1, 9, 0, -1],
        110 => vec![4, 3, 0, 7, 3, 4, 6, 5, 10, -1],
        111 => vec![10, 6, 5, 8, 4, 7, -1],
        112 => vec![9, 6, 5, 9, 11, 6, 9, 8, 11, -1],
        113 => vec![11, 6, 3, 3, 6, 0, 0, 6, 5, 0, 5, 9, -1],
        114 => vec![11, 6, 5, 11, 5, 0, 5, 1, 0, 8, 11, 0, -1],
        115 => vec![11, 6, 3, 6, 5, 3, 5, 1, 3, -1],
        116 => vec![9, 8, 5, 8, 3, 2, 5, 8, 2, 5, 2, 6, -1],
        117 => vec![5, 9, 6, 9, 0, 6, 0, 2, 6, -1],
        118 => vec![1, 6, 5, 2, 6, 1, 3, 0, 8, -1],
        119 => vec![1, 6, 5, 2, 6, 1, -1],
        120 => vec![2, 1, 10, 9, 6, 5, 9, 11, 6, 9, 8, 11, -1],
        121 => vec![9, 0, 1, 3, 11, 2, 5, 10, 6, -1],
        122 => vec![11, 0, 8, 2, 0, 11, 10, 6, 5, -1],
        123 => vec![3, 11, 2, 5, 10, 6, -1],
        124 => vec![1, 8, 3, 9, 8, 1, 5, 10, 6, -1],
        125 => vec![6, 5, 10, 0, 1, 9, -1],
        126 => vec![8, 3, 0, 5, 10, 6, -1],
        127 => vec![6, 5, 10, -1],
        128 => vec![10, 5, 6, -1],
        129 => vec![0, 3, 8, 6, 10, 5, -1],
        130 => vec![10, 5, 6, 9, 1, 0, -1],
        131 => vec![3, 8, 1, 1, 8, 9, 6, 10, 5, -1],
        132 => vec![2, 11, 3, 6, 10, 5, -1],
        133 => vec![8, 0, 11, 11, 0, 2, 5, 6, 10, -1],
        134 => vec![1, 0, 9, 2, 11, 3, 6, 10, 5, -1],
        135 => vec![5, 6, 10, 11, 1, 2, 11, 9, 1, 11, 8, 9, -1],
        136 => vec![5, 6, 1, 1, 6, 2, -1],
        137 => vec![5, 6, 1, 1, 6, 2, 8, 0, 3, -1],
        138 => vec![6, 9, 5, 6, 0, 9, 6, 2, 0, -1],
        139 => vec![6, 2, 5, 2, 3, 8, 5, 2, 8, 5, 8, 9, -1],
        140 => vec![3, 6, 11, 3, 5, 6, 3, 1, 5, -1],
        141 => vec![8, 0, 1, 8, 1, 6, 1, 5, 6, 11, 8, 6, -1],
        142 => vec![11, 3, 6, 6, 3, 5, 5, 3, 0, 5, 0, 9, -1],
        143 => vec![5, 6, 9, 6, 11, 9, 11, 8, 9, -1],
        144 => vec![5, 6, 10, 7, 4, 8, -1],
        145 => vec![0, 3, 4, 4, 3, 7, 10, 5, 6, -1],
        146 => vec![5, 6, 10, 4, 8, 7, 0, 9, 1, -1],
        147 => vec![6, 10, 5, 1, 4, 9, 1, 7, 4, 1, 3, 7, -1],
        148 => vec![7, 4, 8, 6, 10, 5, 2, 11, 3, -1],
        149 => vec![10, 5, 6, 4, 11, 7, 4, 2, 11, 4, 0, 2, -1],
        150 => vec![4, 8, 7, 6, 10, 5, 3, 2, 11, 1, 0, 9, -1],
        151 => vec![1, 2, 10, 11, 7, 6, 9, 5, 4, -1],
        152 => vec![2, 1, 6, 6, 1, 5, 8, 7, 4, -1],
        153 => vec![0, 3, 7, 0, 7, 4, 2, 1, 6, 1, 5, 6, -1],
        154 => vec![8, 7, 4, 6, 9, 5, 6, 0, 9, 6, 2, 0, -1],
        155 => vec![7, 2, 3, 6, 2, 7, 5, 4, 9, -1],
        156 => vec![4, 8, 7, 3, 6, 11, 3, 5, 6, 3, 1, 5, -1],
        157 => vec![5, 0, 1, 4, 0, 5, 7, 6, 11, -1],
        158 => vec![9, 5, 4, 6, 11, 7, 0, 8, 3, -1],
        159 => vec![11, 7, 6, 9, 5, 4, -1],
        160 => vec![6, 10, 4, 4, 10, 9, -1],
        161 => vec![6, 10, 4, 4, 10, 9, 3, 8, 0, -1],
        162 => vec![0, 10, 1, 0, 6, 10, 0, 4, 6, -1],
        163 => vec![6, 10, 1, 6, 1, 8, 1, 3, 8, 4, 6, 8, -1],
        164 => vec![9, 4, 10, 10, 4, 6, 3, 2, 11, -1],
        165 => vec![2, 11, 8, 2, 8, 0, 6, 10, 4, 10, 9, 4, -1],
        166 => vec![11, 3, 2, 0, 10, 1, 0, 6, 10, 0, 4, 6, -1],
        167 => vec![6, 8, 4, 11, 8, 6, 2, 10, 1, -1],
        168 => vec![4, 1, 9, 4, 2, 1, 4, 6, 2, -1],
        169 => vec![3, 8, 0, 4, 1, 9, 4, 2, 1, 4, 6, 2, -1],
        170 => vec![6, 2, 4, 4, 2, 0, -1],
        171 => vec![3, 8, 2, 8, 4, 2, 4, 6, 2, -1],
        172 => vec![4, 6, 9, 6, 11, 3, 9, 6, 3, 9, 3, 1, -1],
        173 => vec![8, 6, 11, 4, 6, 8, 9, 0, 1, -1],
        174 => vec![11, 3, 6, 3, 0, 6, 0, 4, 6, -1],
        175 => vec![8, 6, 11, 4, 6, 8, -1],
        176 => vec![10, 7, 6, 10, 8, 7, 10, 9, 8, -1],
        177 => vec![3, 7, 0, 7, 6, 10, 0, 7, 10, 0, 10, 9, -1],
        178 => vec![6, 10, 7, 7, 10, 8, 8, 10, 1, 8, 1, 0, -1],
        179 => vec![6, 10, 7, 10, 1, 7, 1, 3, 7, -1],
        180 => vec![3, 2, 11, 10, 7, 6, 10, 8, 7, 10, 9, 8, -1],
        181 => vec![2, 9, 0, 10, 9, 2, 6, 11, 7, -1],
        182 => vec![0, 8, 3, 7, 6, 11, 1, 2, 10, -1],
        183 => vec![7, 6, 11, 1, 2, 10, -1],
        184 => vec![2, 1, 9, 2, 9, 7, 9, 8, 7, 6, 2, 7, -1],
        185 => vec![2, 7, 6, 3, 7, 2, 0, 1, 9, -1],
        186 => vec![8, 7, 0, 7, 6, 0, 6, 2, 0, -1],
        187 => vec![7, 2, 3, 6, 2, 7, -1],
        188 => vec![8, 1, 9, 3, 1, 8, 11, 7, 6, -1],
        189 => vec![11, 7, 6, 1, 9, 0, -1],
        190 => vec![6, 11, 7, 0, 8, 3, -1],
        191 => vec![11, 7, 6, -1],
        192 => vec![7, 11, 5, 5, 11, 10, -1],
        193 => vec![10, 5, 11, 11, 5, 7, 0, 3, 8, -1],
        194 => vec![7, 11, 5, 5, 11, 10, 0, 9, 1, -1],
        195 => vec![7, 11, 10, 7, 10, 5, 3, 8, 1, 8, 9, 1, -1],
        196 => vec![5, 2, 10, 5, 3, 2, 5, 7, 3, -1],
        197 => vec![5, 7, 10, 7, 8, 0, 10, 7, 0, 10, 0, 2, -1],
        198 => vec![0, 9, 1, 5, 2, 10, 5, 3, 2, 5, 7, 3, -1],
        199 => vec![9, 7, 8, 5, 7, 9, 10, 1, 2, -1],
        200 => vec![1, 11, 2, 1, 7, 11, 1, 5, 7, -1],
        201 => vec![8, 0, 3, 1, 11, 2, 1, 7, 11, 1, 5, 7, -1],
        202 => vec![7, 11, 2, 7, 2, 9, 2, 0, 9, 5, 7, 9, -1],
        203 => vec![7, 9, 5, 8, 9, 7, 3, 11, 2, -1],
        204 => vec![3, 1, 7, 7, 1, 5, -1],
        205 => vec![8, 0, 7, 0, 1, 7, 1, 5, 7, -1],
        206 => vec![0, 9, 3, 9, 5, 3, 5, 7, 3, -1],
        207 => vec![9, 7, 8, 5, 7, 9, -1],
        208 => vec![8, 5, 4, 8, 10, 5, 8, 11, 10, -1],
        209 => vec![0, 3, 11, 0, 11, 5, 11, 10, 5, 4, 0, 5, -1],
        210 => vec![1, 0, 9, 8, 5, 4, 8, 10, 5, 8, 11, 10, -1],
        211 => vec![10, 3, 11, 1, 3, 10, 9, 5, 4, -1],
        212 => vec![3, 2, 8, 8, 2, 4, 4, 2, 10, 4, 10, 5, -1],
        213 => vec![10, 5, 2, 5, 4, 2, 4, 0, 2, -1],
        214 => vec![5, 4, 9, 8, 3, 0, 10, 1, 2, -1],
        215 => vec![2, 10, 1, 4, 9, 5, -1],
        216 => vec![8, 11, 4, 11, 2, 1, 4, 11, 1, 4, 1, 5, -1],
        217 => vec![0, 5, 4, 1, 5, 0, 2, 3, 11, -1],
        218 => vec![0, 11, 2, 8, 11, 0, 4, 9, 5, -1],
        219 => vec![5, 4, 9, 2, 3, 11, -1],
        220 => vec![4, 8, 5, 8, 3, 5, 3, 1, 5, -1],
        221 => vec![0, 5, 4, 1, 5, 0, -1],
        222 => vec![5, 4, 9, 3, 0, 8, -1],
        223 => vec![5, 4, 9, -1],
        224 => vec![11, 4, 7, 11, 9, 4, 11, 10, 9, -1],
        225 => vec![0, 3, 8, 11, 4, 7, 11, 9, 4, 11, 10, 9, -1],
        226 => vec![11, 10, 7, 10, 1, 0, 7, 10, 0, 7, 0, 4, -1],
        227 => vec![3, 10, 1, 11, 10, 3, 7, 8, 4, -1],
        228 => vec![3, 2, 10, 3, 10, 4, 10, 9, 4, 7, 3, 4, -1],
        229 => vec![9, 2, 10, 0, 2, 9, 8, 4, 7, -1],
        230 => vec![3, 4, 7, 0, 4, 3, 1, 2, 10, -1],
        231 => vec![7, 8, 4, 10, 1, 2, -1],
        232 => vec![7, 11, 4, 4, 11, 9, 9, 11, 2, 9, 2, 1, -1],
        233 => vec![1, 9, 0, 4, 7, 8, 2, 3, 11, -1],
        234 => vec![7, 11, 4, 11, 2, 4, 2, 0, 4, -1],
        235 => vec![4, 7, 8, 2, 3, 11, -1],
        236 => vec![9, 4, 1, 4, 7, 1, 7, 3, 1, -1],
        237 => vec![7, 8, 4, 1, 9, 0, -1],
        238 => vec![3, 4, 7, 0, 4, 3, -1],
        239 => vec![7, 8, 4, -1],
        240 => vec![11, 10, 8, 8, 10, 9, -1],
        241 => vec![0, 3, 9, 3, 11, 9, 11, 10, 9, -1],
        242 => vec![1, 0, 10, 0, 8, 10, 8, 11, 10, -1],
        243 => vec![10, 3, 11, 1, 3, 10, -1],
        244 => vec![3, 2, 8, 2, 10, 8, 10, 9, 8, -1],
        245 => vec![9, 2, 10, 0, 2, 9, -1],
        246 => vec![8, 3, 0, 10, 1, 2, -1],
        247 => vec![2, 10, 1, -1],
        248 => vec![2, 1, 11, 1, 9, 11, 9, 8, 11, -1],
        249 => vec![11, 2, 3, 9, 0, 1, -1],
        250 => vec![11, 0, 8, 2, 0, 11, -1],
        251 => vec![3, 11, 2, -1],
        252 => vec![1, 8, 3, 9, 8, 1, -1],
        253 => vec![1, 9, 0, -1],
        254 => vec![8, 3, 0, -1],
        255 => vec![-1],
        _ => vec![-1],
    }
}
