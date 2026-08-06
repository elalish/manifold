module noop() {}

module children_test() {
	children() noop();
}
children_test() noop();


surface("../assets/smiley.png") noop();

text("Hello World!", 26, font = "Liberation Sans:style=Regular") noop();
