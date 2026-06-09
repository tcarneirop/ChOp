module queens_node_evaluation{

inline proc stillLegal(board, r) {
		var safe = true;
		const base = board[r];
		for (i, rev_i, offset) in zip(0..<r, 0..<r by -1, 1..r) {
			safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
						(board[rev_i] == base+offset)));
		}
		return safe;
	}

}

