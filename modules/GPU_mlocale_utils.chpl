module GPU_mlocale_utils{

	use CTypes;
	proc GPU_mlocale_get_gpu_load(const survivors: c_uint, const gpu_id:c_int, const num_gpus: c_int ): c_uint{

		var local_load: c_uint = survivors/(num_gpus:c_uint);

		if gpu_id == (num_gpus - 1){
			if survivors % num_gpus then
				local_load += (survivors % (num_gpus:c_uint));
		}

		return local_load;
	}////////////

	proc GPU_mlocale_get_starting_point(const survivors: c_uint, const gpu_id:c_uint,
		const num_gpus: c_uint, const cpu_load: c_uint ): c_uint{

		var cpu_end: c_uint = if cpu_load>0 then (cpu_load+1) else 0;

		return ((gpu_id*(survivors/num_gpus))+cpu_end);
	}////////////


	proc GPU_mlocale_get_starting_point(survivors: c_uint): c_int{
			return (here.id*(survivors/numLocales)):c_int;
	}////////////


	proc GPU_mlocale_get_locale_load(survivors: c_uint): c_uint{

		var l_numLocales: c_uint = numLocales:c_uint;

		var local_load: c_uint = survivors/l_numLocales;

		if here.id == (numLocales - 1){
			if survivors % l_numLocales then
				local_load += (survivors % l_numLocales);
		}

		return local_load;
	}////////////qq



	proc GPU_mlocale_print_load_position(const survivors: c_uint){
		writeln("\nHello, I'm locale ", here.id, " and my load is: ", GPU_mlocale_get_locale_load(survivors),
			" my initial position is: ", GPU_mlocale_get_starting_point(survivors), " and my final position is: ",
			GPU_mlocale_get_starting_point(survivors)+GPU_mlocale_get_locale_load(survivors)-1);
	}/////////////


	proc GPU_mlocale_get_GPU(const mode: string, const real_number_computers: int,
		const num_gpus_computer: int, const locale_id: int): int{//is gpu, gpu id, real node id, real node name


		var gpu_status: (bool, int, int, string); //is going to return the gpu status, and the GPU id on the computer node.
		var gpu_id: int;
		var num_locales_computer = if mode == "cpugpu" then num_gpus_computer+1 else num_gpus_computer;
		var real_computer: string;
		var real_computer_id: int;

		if mode == "cpugpu" then {
			if locale_id % num_locales_computer == 0 then{
				gpu_status[0] = false;
				gpu_status[1] = -1;
			}
			else {
				gpu_status[0] = true;
				gpu_id = (locale_id % num_locales_computer) -1;
				gpu_status[1] = gpu_id;
			}//

			gpu_status[2] = locale_id/num_locales_computer*num_locales_computer; //real computer
			gpu_status[3] = Locales[gpu_status[2]].name;
		}
		else{//mgpu
			gpu_status[0] = true;
			gpu_id = (locale_id % num_locales_computer);
			gpu_status[1] = gpu_id;
		}
		return gpu_status[1];
	}///


	proc GPU_mlocale_get_locale_role(const mode: string,real_number_computers: int,
	num_gpus_computer: int, locale_id: int): (bool, int, int, string) {//is gpu, gpu id, real node id, real node name

		var gpu_status: (bool, int, int, string); //is going to return the gpu status, and the GPU id on the computer node.
		var gpu_id: int;
		var num_locales_computer = if mode == "cpugpu" then num_gpus_computer+1 else num_gpus_computer;
		var real_computer: string;
		var real_computer_id: int;

		writeln(mode);

		if mode == "cpugpu" then {
			if locale_id % num_locales_computer == 0 then{
				gpu_status[0] = false;
				gpu_status[1] = -1;
			}
			else {
				gpu_status[0] = true;
				gpu_id = (locale_id % num_locales_computer) -1;
				gpu_status[1] = gpu_id;
			}//

			gpu_status[2] = locale_id/num_locales_computer*num_locales_computer; //real computer
			gpu_status[3] = Locales[gpu_status[2]].name;
		}
		else{///mgpu
			gpu_status[0] = true;
			gpu_id = (locale_id % num_locales_computer);
			gpu_status[1] = gpu_id;
		}

		return gpu_status;
	}///

	/*
		This fuction is used when the GPUs of the locale are used along with the CPUs;
	*/
	proc GPU_mlocale_number_locales_check(const mode: string, const flag_coordinated: int){

		if(mode == "cpugpu"){


			writeln("\nReal number of computers: ", Locales.size/2+flag_coordinated,"\nNumber of locales: ", Locales.size,
				"\nCoordinated: ", flag_coordinated);

			if (((Locales.size - flag_coordinated) % 2 )) then
				halt("\n###### Wrong CPU-GPU nl parameter ######");
			else
				writeln("\n### !!! Num locales for the GPU-CPU search is OK !!! ###\n");
		}
		else
			writeln("\n### !!! Num locales for the GPU-CPU search is OK !!! ###\n");

		for loc in Locales do{
            on loc do{
                if(here.id == 0 && flag_coordinated==true){
                    writeln("Coordinator: ", here.id," - ", here.name,"\n");

                }
                else{
                    writeln("Locale: ", here.name," - " ,here.id,"\n\tRole: ", (here.id-flag_coordinated)%2);
                }
            }///
        }///

	}////

}/////module
