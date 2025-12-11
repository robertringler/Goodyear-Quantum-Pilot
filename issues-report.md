## Remaining Code Quality Issues
### Ruff Linting
warning: The top-level linter settings are deprecated in favour of their counterparts in the `lint` section. Please update the following options in `goodyear_quantum_pilot/pyproject.toml`:
  - 'ignore' -> 'lint.ignore'
  - 'select' -> 'lint.select'
  - 'isort' -> 'lint.isort'
  - 'per-file-ignores' -> 'lint.per-file-ignores'
1303	UP006 	[ ] non-pep585-annotation
 345	      	[ ] invalid-syntax
 200	N806  	[ ] non-lowercase-variable-in-function
 105	UP045 	[ ] non-pep604-annotation-optional
  69	F841  	[ ] unused-variable
  69	W291  	[ ] trailing-whitespace
  42	E402  	[ ] module-import-not-at-top-of-file
  36	N803  	[ ] invalid-argument-name
  35	W293  	[ ] blank-line-with-whitespace
  32	B007  	[ ] unused-loop-control-variable
  32	F821  	[ ] undefined-name
  20	SIM108	[ ] if-else-block-instead-of-if-exp
  19	ARG002	[ ] unused-method-argument
  16	UP007 	[ ] non-pep604-annotation-union
  14	SIM102	[ ] collapsible-if
  10	B905  	[ ] zip-without-explicit-strict
   8	B904  	[ ] raise-without-from-inside-except
   7	C401  	[ ] unnecessary-generator-set
   7	F811  	[ ] redefined-while-unused
   7	SIM103	[ ] needless-bool
   6	F401  	[ ] unused-import
   5	N802  	[ ] invalid-function-name
   4	ARG001	[ ] unused-function-argument
   3	N999  	[ ] invalid-module-name
   3	SIM105	[ ] suppressible-exception
   2	B006  	[ ] mutable-argument-default
   2	C416  	[ ] unnecessary-comprehension
   2	N818  	[ ] error-suffix-on-exception-name
   2	SIM110	[ ] reimplemented-builtin
   2	SIM118	[ ] in-dict-keys
   1	B011  	[ ] assert-false
   1	C400  	[ ] unnecessary-generator-list
   1	C414  	[ ] unnecessary-double-cast-or-process
   1	E731  	[ ] lambda-assignment
   1	I001  	[*] unsorted-imports
   1	SIM113	[ ] enumerate-for-loop
Found 2413 errors.
[*] 1 fixable with the `--fix` option (1677 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api is not a valid Python package name
Type checking completed
