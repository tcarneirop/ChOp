

module statistics{

    use Math;

    proc statistics_all_locales_init_explored_tree(ref tree_each_locale: [] uint(64)){
        writeln("# Starting the metrics on each locale. #");
        forall a in tree_each_locale do
            a = 0:uint(64);
    }/////

    proc statistics_tree_statistics(ref tree_each_locale: [] uint(64), const tree_size: uint(64)){

        var local_tree_each_locale: [0..numLocales-1] uint(64);
        var biggest: uint(64)  = 0;
        var smallest: uint(64) = max(uint(64));
        var value:    uint(64) = 0;
        var biggest_idx:   int = 0;
        var smallest_idx:   int = 0;

        forall (a,b) in zip(local_tree_each_locale, tree_each_locale) do
            a = b;

        writeln("\n### Locales statistics (in % of the total tree size): ###\n");

        for i in 0..numLocales-1 do{
            value = local_tree_each_locale[i];
            writeln("\tLocale ", i, ": ", value, " - ", (value:real(64)/tree_size:real(64)*100:real(64)):real(64), "%" );
        }

        for i in 0..numLocales-1 do{
            //writeln("i: ", i, " tree: ", local_tree_each_locale[i] );
            if(local_tree_each_locale[i]>0){
                //writeln("i: ", i, " tree: ", local_tree_each_locale[i] );
                if (local_tree_each_locale[i] > biggest) {
                    biggest_idx = i;
                    biggest = local_tree_each_locale[i];
                }
            }
        }

        for i in 0..numLocales-1 do{
            //writeln("i: ", i, " tree: ", local_tree_each_locale[i] );
            if(local_tree_each_locale[i]>0){
                //writeln("i: ", i, " tree: ", local_tree_each_locale[i] );
                if (local_tree_each_locale[i] < smallest){
                    smallest_idx = i;
                    //writeln("New smallest: from ", smallest, " to ", local_tree_each_locale[i], ".");
                    smallest = local_tree_each_locale[i];
                }
            }
        }

        writeln("\nBiggest subtree:\n\tLocale ",  biggest_idx, " - ", local_tree_each_locale[biggest_idx] );
        writeln("\nSmallest subtree:\n\tLocale ", smallest_idx, " - ", local_tree_each_locale[smallest_idx] );
        writeln("\nRatio biggest/smallest: ", (local_tree_each_locale[biggest_idx]:real(64))/(local_tree_each_locale[smallest_idx]:real(64)) );
    }/////

}
