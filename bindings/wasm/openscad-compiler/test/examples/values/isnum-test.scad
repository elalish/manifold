// a number is a number
echo(is_num(0.1));
echo(is_num(1));
echo(is_num(10));

// inf is a number
echo(is_num(+1/0)); //+inf
echo(is_num(-1/0)); //-inf

// nan is not a number
echo(is_num(0/0)); //nan
echo(is_num((1/0)/(1/0)));  //nan

// resulting in false
echo(is_num([]));
echo(is_num([1]));
echo(is_num("test"));
echo(is_num(false));
echo(is_num(undef)); 

// resulting in warnings
echo(is_num(1,2,3)); 
echo(is_num()); 