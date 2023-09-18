module queens_node_evaluation{

	inline proc stillLegal(board: [] int(8), const r: int  ): bool{

		var ld:  int(8);
		var rd:  int(8);

		// var example_range = 0..10;

		for i in 0..(r-1) do {
			if board[i] == board[r] then
				return false;
		}

		ld = board[r];  //left diagonal columns
	    rd = board[r];  // right diagonal columns

	    for j in 0..(r-1) by -1 do{ // for ( i = r-1; i >= 0; --i) {
	    	ld -= 1;
	    	rd += 1;
	      	if board[j] == ld || board[j] == rd then
	      		return false;
	    }


		return true;
	}

}

