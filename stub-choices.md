While stubbing the library, some decisions have been made that lead _type-checked_ code to be stricter than just _runtime valid_ code. This document tries to capture the reason behind those choices and the spirit they were made in.

1. **Using enums**

SCIP has many C enums, which means they are really just numbers. PySCIPOpt
includes their values as class variables which are to be treated as constants.
But in the end, they are still just numbers and mistakes are highly possible.
The stubs makes all of those into Python enums. The consequence is that the
values cannot be used directly, one must use the enum attribute access.
This is the best interest of the developer, as the enum values were never meant
to be used on their own.

2. **Plugin return types**

Nearly all plugin methods that return a value must return it as a dictionary.
The relevant keys are checked by the Cython layer, and the appropriate reference
is set such that the C code can use the return value.
These dictionaries are modelled in the stubs using `TypedDict`s. This enables the
developer to know which keys are relevant, and what type the values should be.
In particular, the `result` key is often used to convey information about the
status of the plugin's operations. Its value should be one the keys in the
`SCIP_RESULT` enum, but not all keys are valid. In the stubs, a restricted list
of keys are given which match the keys accepted by the underlying C code.
An important fact is that all dictionary values are optional at runtime with
defaults that usually correspond to the plugin doing nothing.
In the stubs, not all keys are optional. In many cases, it was decided to make
the keys required to force the developer to be explicit. For instance, this was
done for the result keys which would default to `DIDNOTRUN` at runtime. To get
correctly type-checked code, the plugin methods must return
`{"result": SCIP_RESULT.DIDNOTRUN}`.

3. **Allowed numeric values**

Many inputs to operations on `Expr` go through the `_is_number` check which just
calls `float` on the input. This allows a lot of things to work at runtime.
For instance, decimals, fractions and even strings can be valid.
However, allowing any string is type-unsafe, so it is not included in the stubs.
For the rest, `SupportsFloat` works well. The issue is that _all_ NumPy arrays
satisfy this protocol, meaning any array is allowed anywhere a scalar is accepted.
To avoid that, it was finally decided to only allow `float` in those places, and
it is up to the user to convert the inputs to floats before using operations
on `Expr`.
