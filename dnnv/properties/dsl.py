import ast
import sys

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from . import base
from .context import Context
from .visitors import ExpressionVisitor


class PropertyParserError(Exception):
    pass


class Py2PropertyTransformer(ast.NodeTransformer):
    def __init__(self):
        self._ssa_ids = defaultdict(int)
        self._lambda_aliases = set()
        self.visiting_symbol_name = None
        self.symbolic_funcs = set()
        self.visited_symbolic_funcs = set()

    def _ssa(self, name):
        ssa_id = self._ssa_ids[name]
        self._ssa_ids[name] += 1
        return f"{name}{ssa_id}"

    def visit_Assign(self, node: ast.Assign):
        if any(not isinstance(target, ast.Name) for target in node.targets):
            raise PropertyParserError(
                "Assigning to non-identifiers is not currently supported"
            )
        if isinstance(node.value, ast.Lambda):
            assert isinstance(node.targets[0], ast.Name)
            self._lambda_aliases.add(node.targets[0].id)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        func = self.visit(node.func)
        # print(f'visiting call {func.id}, symbol name: {self.visiting_symbol_name}')
        args = [self.visit(arg) for arg in node.args]
        kwargs = [self.visit(keyword) for keyword in node.keywords]

        if isinstance(func, ast.Name):
            if func.id in ["Forall", "Exists"]:
                if len(args) != 2:
                    raise ValueError("%s takes 2 arguments." % func.id)
                if len(kwargs) != 0:
                    raise ValueError("%s does not take keyword arguments." % func.id)
                if not isinstance(args[0], ast.Name):
                    raise PropertyParserError(
                        "The first argument to %s must be a variable name." % func.id
                    )
                symbol_name = args[0]
                self.visiting_symbol_name = symbol_name.id
                args = [self.visit(arg) for arg in node.args]
                sym_func = ast.Name("Symbol", ast.Load(), **attributes)
                sym_func_args = [
                    ast.Str(self._ssa(symbol_name.id), **attributes)
                ]  # type: List[ast.AST]
                variable = ast.Call(sym_func, sym_func_args, [], **attributes)

                orig_expr = args[1]
                if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
                    lambda_args = ast.arguments(
                        [],
                        [ast.arg(symbol_name.id, None, **attributes)],
                        None,
                        [],
                        [],
                        None,
                        [],
                    )
                elif sys.version_info.major >= 3:
                    lambda_args = ast.arguments(
                        [ast.arg(symbol_name.id, None, **attributes)],
                        None,
                        [],
                        [],
                        None,
                        [],
                    )
                else:
                    raise PropertyParserError(
                        "Currently only Python versions 3.7+ are supported."
                    )
                lambda_func = ast.Lambda(lambda_args, orig_expr, **attributes)
                new_expr = ast.Call(lambda_func, [variable], [], **attributes)

                new_args = [variable, lambda_func]
                new_node = ast.Call(func, new_args, [], **attributes)

                self.visiting_symbol_name = None
                return new_node
            elif hasattr(self, "local_func_name") and func.id in self.local_func_name:
                if self.visiting_symbol_name is not None and len(args) > 0 and isinstance(args[0], ast.Name) and args[0].id == self.visiting_symbol_name:
                    self.symbolic_funcs.add(func.id)
                return node
            else:
                value = base.__dict__.get(func.id, None)
                if (
                    value is not None
                    and isinstance(value, type)
                    and issubclass(value, base.Expression)
                ):
                    return ast.Call(func, args, kwargs, **attributes)
                if func.id in self._lambda_aliases:
                    return ast.Call(func, args, kwargs, **attributes)

        make_func = ast.Name("_symbol_from_callable", ast.Load(), **attributes)
        func_expr = ast.Call(make_func, [func], [], **attributes)
        new_node = ast.Call(func_expr, args, kwargs, **attributes)
        return new_node

    def visit_Compare(self, node: ast.Compare):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if len(node.ops) == 1:
            return self.generic_visit(node)
        comparisons = []
        left = self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            right = self.visit(right)
            if isinstance(op, ast.LtE):
                func = ast.Name("LessThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Lt):
                func = ast.Name("LessThan", ast.Load(), **attributes)
            elif isinstance(op, ast.GtE):
                func = ast.Name("GreaterThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Gt):
                func = ast.Name("GreaterThan", ast.Load(), **attributes)
            else:
                raise ValueError("Unsupported comparison function: %s" % op)
            comparisons.append(ast.Call(func, [left, right], [], **attributes))
            left = right
        and_func = ast.Name("And", ast.Load(), **attributes)
        new_node = ast.Call(and_func, comparisons, [], **attributes)
        return new_node

    def visit_Ellipsis(self, node: ast.Ellipsis):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_NameConstant(self, node: ast.NameConstant):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Num(self, node: ast.Num):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Str(self, node: ast.Str):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Dict(self, node: ast.Dict):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        invalid_keys = [
            key
            for key in node.keys
            if not isinstance(key, (ast.NameConstant, ast.Num, ast.Str))
        ]
        invalid_values = [
            value
            for value in node.values
            if not isinstance(value, (ast.NameConstant, ast.Num, ast.Str))
        ]
        if len(invalid_keys) > 0:
            raise PropertyParserError(
                "We do not currently support definition of dicts containing non-primitive keys."
            )
        if len(invalid_values) > 0:
            raise PropertyParserError(
                "We do not currently support definition of dicts containing non-primitive values."
            )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def _ensure_primitive(self, expr):
        if isinstance(
            expr,
            (
                ast.Dict,
                ast.List,
                ast.NameConstant,
                ast.Num,
                ast.Set,
                ast.Str,
                ast.Tuple,
            ),
        ):
            return True
        elif isinstance(expr, ast.UnaryOp) and isinstance(
            expr.operand, (ast.NameConstant, ast.Num, ast.Str)
        ):
            return True
        return False

    def visit_List(self, node: ast.List):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        for expr in node.elts:
            if not self._ensure_primitive(expr):
                raise PropertyParserError(
                    "We do not currently support definition of lists containing non-primitive types."
                )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Set(self, node: ast.Set):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        for expr in node.elts:
            if not self._ensure_primitive(expr):
                raise PropertyParserError(
                    "We do not currently support definition of sets containing non-primitive types."
                )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Tuple(self, node: ast.Tuple):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        for expr in node.elts:
            if not self._ensure_primitive(expr):
                raise PropertyParserError(
                    "We do not currently support definition of tuples containing non-primitive types."
                )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Slice(self, node: ast.Slice):
        attributes = {"lineno": 0, "col_offset": 0}

        start = (
            self.visit(node.lower)
            if node.lower is not None
            else ast.NameConstant(None, **attributes)
        )
        stop = (
            self.visit(node.upper)
            if node.upper is not None
            else ast.NameConstant(None, **attributes)
        )
        step = (
            self.visit(node.step)
            if node.step is not None
            else ast.NameConstant(None, **attributes)
        )

        slice_func = ast.Name("Slice", ast.Load(), **attributes)
        new_node = ast.Call(slice_func, [start, stop, step], [], **attributes)
        index_node = ast.Index(new_node)
        return index_node

    def visit_ExtSlice(self, node: ast.ExtSlice):
        dims = [
            dim
            for dim in node.dims
            if not isinstance(
                dim, (ast.NameConstant, ast.Num, ast.Str, ast.Slice, ast.Index)
            )
        ]
        if len(dims) > 0:
            raise PropertyParserError(
                "We do not currently support definition of slices containing non-primitive types."
            )
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Index(self, node: ast.Index):
        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await):
        raise PropertyParserError("We do not support await expressions.")

    def visit_Yield(self, node: ast.Yield):
        raise PropertyParserError("We do not support yield expressions.")

    def visit_YieldFrom(self, node: ast.YieldFrom):
        raise PropertyParserError("We do not support yield from expressions.")

    def visit_IfExp(self, node: ast.IfExp):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        ite_func = ast.Name("IfThenElse", ast.Load(), **attributes)
        new_node = ast.Call(ite_func, [test, body, orelse], [], **attributes)
        return new_node

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise PropertyParserError("We do not currently support generator expressions.")

    def visit_DictComp(self, node: ast.DictComp):
        raise PropertyParserError("We do not currently support dict comprehensions.")

    def visit_ListComp(self, node: ast.ListComp):
        raise PropertyParserError("We do not currently support list comprehensions.")

    def visit_SetComp(self, node: ast.SetComp):
        raise PropertyParserError("We do not currently support set comprehensions.")

    def visit_Starred(self, node: ast.Starred):
        raise PropertyParserError("We do not currently support starred expressions.")

class O_Py2PropertyTransformer(Py2PropertyTransformer):

    def visit_FunctionDef(self, node: ast.FunctionDef):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if not hasattr(self, "local_func_name"):
            self.local_func_name = set()
        self.local_func_name.add(node.name)
        #print("local func", repr(self.local_func_name))
        # FIXME: check assumptions

        if node.name not in self.symbolic_funcs:
            return node
        else:
            # symbolic func
            # remove it from revisit list
            self.symbolic_funcs.remove(node.name)
            # mark it as visited to prevent loop (foo->bar->foo->bar->...)
            self.visited_symbolic_funcs.add(node.name)

            symbol_name = self.visit(node.args.args[0])
            self.visiting_symbol_name = symbol_name.arg
            new_body = []
            for stmt in node.body:
                new_stmt = self.visit(stmt)
                new_body.append(new_stmt)
            self.visiting_symbol_name = None

            return ast.FunctionDef(name = node.name,
                                   args = node.args,
                                   body = new_body,
                                   decorator_list = node.decorator_list,
                                   returns = node.returns,
                                   #type_comment = node.type_comment,
                                   **attributes)

    def visit_List(self, node: ast.List):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}

        new_node_elts = []
        raise_exception = False
        for expr in node.elts:
            if not self._ensure_primitive(expr):
                # we allow local function as well
                if isinstance(expr, ast.Call) and \
                   hasattr(self, "local_func_name") and \
                   expr.func.id in self.local_func_name:
                    # okay
                    pass
                else:
                    raise_exception = True

            if raise_exception:
                raise PropertyParserError(
                    "We do not currently support definition of lists containing non-primitive types. [%s]" % (repr(expr))
                )
            # visit each elts
            new_node_elts.append(self.visit(expr))

        # create new list node with element visited
        new_node = ast.List(elts=new_node_elts,
                            ctx=node.ctx,
                            **attributes)

        # wrap a constant call
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [new_node], [], **attributes)

    def visit_ListComp(self, node: ast.ListComp):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        # [elt for gen.target in gen.iter]
        # assumptions:
        #  - gen.target is ast.Name
        #  - gen.iter is either runnable function or some concrete iterable obj
        elt = self.visit(node.elt)

        if len(node.generators) != 1:
            raise PropertyParserError("No support for more than one generator in ListComp.")
        generator = node.generators[0]
        if len(generator.ifs) > 0:
            raise PropertyParserError("No support for generator.ifs in ListComp.")
        if generator.is_async != 0:
            raise PropertyParserError("No support for generator.is_async!=0 in ListComp.")
        if not isinstance(generator.target, ast.Name):
            raise PropertyParserError("No support for non-primitive gen.target in ListComp.")
        # FIXME: make sure these are not symbolic vars
        if not (
            isinstance(generator.iter, ast.Name) or # a iterable
            isinstance(generator.iter, ast.Call)    # function call
        ):
            raise PropertyParserError("No support for gen.iter (%s) in ListComp." % type(generator.iter))

        g_target = self.visit(generator.target)
        g_iter = self.visit(generator.iter)

        return ast.ListComp(elt=elt, generators=[
            ast.comprehension(target=g_target,
                              iter=g_iter,
                              ifs=[], is_async=0)], **attributes)


    def visit_Subscript(self, node: ast.Subscript):
        # FIXME:
        # expr = self.visit(expression.expr1)[self.visit(expression.expr2)]
        # return expr
        #print(ast.dump(node))
        return node

    def revisit(self, body: List[ast.stmt]):
        # revisit the FunctionDefs that contain symbols
        # keep revisiting until there is no new symbolic functions found
        while len(self.symbolic_funcs) != 0:
            # print(f'----- ----- ----- revisit {self.symbolic_funcs} ----- ----- -----')
            for i in range(len(body)):
                if isinstance(body[i], ast.FunctionDef) and body[i].name in self.symbolic_funcs:
                    body[i] = self.visit(body[i])


class LimitQuantifiers(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.at_top_level = True
        self.top_level_quantifier = None

    def __call__(self, phi):
        self.at_top_level = True
        self.top_level_quantifier = None
        if isinstance(phi, base.Quantifier):
            self.top_level_quantifier = phi.__class__
        self.visit(phi)

    def generic_visit(self, expr):
        self.at_top_level = False
        super().generic_visit(expr)

    def visit_Exists(self, expr):
        if not self.at_top_level:
            raise PropertyParserError("Quantifiers are only allowed at the top level.")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type."
            )
        self.visit(expr.expression)

    def visit_Forall(self, expr):
        if not self.at_top_level:
            raise PropertyParserError("Quantifiers are only allowed at the top level.")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type."
            )
        self.visit(expr.expression)


class SymbolFactory(dict):
    def __getitem__(self, item):
        if item not in self:
            assert isinstance(item, str)
            super().__setitem__(item, base.Symbol(item))
        return super().__getitem__(item)


def parse_cli(phi: base.Expression, args):
    import argparse

    parser = argparse.ArgumentParser()

    parameters = phi.parameters
    for parameter in parameters:
        default = parameter.default
        if isinstance(default, base.Expression) and default.is_concrete:
            default = default.value
        parser.add_argument(
            f"--prop.{parameter.name}", type=parameter.type, default=default
        )
    known_args, unknown_args = parser.parse_known_args(args)
    if args is not None:
        args.clear()
        args.extend(unknown_args)
    for parameter in parameters:
        parameter_value = getattr(known_args, f"prop.{parameter.name}")
        if parameter_value is None:
            raise PropertyParserError(
                f"No argument was provided for parameter '{parameter.name}'. "
                f"Try adding a command line argument '--prop.{parameter.name}'."
            )
        parameter.concretize(parameter_value)

def parse(path: Path, args: Optional[List[str]] = None) -> base.Expression:
    with open(path, "r") as f:
        module = ast.parse(f.read())
    for node in module.body[:-1]:
        if not isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom, ast.FunctionDef)) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)
        ):
            raise PropertyParserError(
                f"Unsupported structure in property (line {node.lineno}): {node}"
            )
    property_node = module.body[-1]
    if not isinstance(property_node, ast.Expr):
        raise PropertyParserError()

    # cheng-hack: see origin below
    #module = Py2PropertyTransformer().visit(module)

    my_transformer = O_Py2PropertyTransformer()
    for i in range(len(module.body)):
        # convert (1) the spec in the end, (2) function with "symbolic" decorator
        if (i == len(module.body) - 1) \
                or isinstance(module.body[i], ast.FunctionDef):
            module.body[i] = my_transformer.visit(module.body[i])

    my_transformer.revisit(module.body)

    attributes = {
        "lineno": property_node.lineno,
        "col_offset": property_node.col_offset,
    }
    module.body[-1] = ast.Assign(
        [ast.Name("phi", ast.Store(), **attributes)], property_node.value, **attributes
    )

    global_dict = SymbolFactory()
    global_dict.update(globals()["__builtins__"])
    global_dict.update(base.__dict__)

    with Context():
        code = compile(module, filename=path.name, mode="exec")
        exec(code, global_dict)
        phi = global_dict["phi"]
        LimitQuantifiers()(phi)

        phi = phi.propagate_constants()
        parse_cli(phi, args)

    return phi
