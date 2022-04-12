module concurrency{

	use CTypes;

	proc min(a: c_int, b: c_int): c_int{

		if a<b then
			return a;
		else
			return b;
	}



	proc concurrency_minExchange(local_upper_bound: c_int, global_upper_bound: atomic c_int){

        while true {
	        var curMin = global_upper_bound.read();
	        if local_upper_bound >= curMin then
	            break;
	        global_upper_bound.compareExchangeWeak(curMin, local_upper_bound);
        }//minExchange
	}

	proc new_concurrency_minExchange(local_upper_bound: c_int, ref global_upper_bound: atomic c_int): c_int{

		var curMin: c_int;
        while true {
	        curMin = global_upper_bound.read();
	        if local_upper_bound >= curMin then{
	        	break;
	        }
	        global_upper_bound.compareExchangeWeak(curMin, local_upper_bound);
        }//minExchange
        return curMin;
	}


	proc concurrency_mlocale_minExchange(incumbent: c_int, ref local_upper_bound: atomic c_int, ref global_upper_bound: atomic c_int): c_int{

		var curMin, g, l: c_int;

		g = new_concurrency_minExchange(incumbent, global_upper_bound);

		curMin = min(incumbent, g);

		l = new_concurrency_minExchange(curMin, local_upper_bound);

		curMin = min(l,curMin);

        // while true {

	       //  g = global_upper_bound.read();
	       //  l = local_upper_bound.read();

	       //  curMin = min(g,l);

	       //  if incumbent >= curMin then{
	       //  	break;
	       //  }
	       //  global_upper_bound.compareExchangeWeak(curMin, incumbent);
	       //  local_upper_bound.compareExchangeWeak(curMin, incumbent);

        // }//minExchange


        return curMin;
	}


}
