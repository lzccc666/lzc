"""当前论文公式的显式语义树；离线MathML和可编辑Word OMML同源导出。

仅转换逐项核对过的原稿表达式；遇到新增表达式直接报错，避免静默损坏数学含义。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def text(value, *, bold=False, normal=False):
    return ('text', value, bold, normal)


def sequence(*items):
    return ('seq', items)


def sub(base, index):
    return ('sub', base, text(index, normal=True))


def sup(base, index):
    return ('sup', base, text(index, normal=True))


def matrix(*rows):
    return ('delim', '[', ']', ('matrix', rows))


P = text('p', bold=True)
PC, PW = sub(P, 'c'), sub(P, 'w')
K, Z = text('K'), text('z')
TWC = sub(text('T'), 'wc')


def key(latex):
    return re.sub(r'\s+', '', latex)


REGISTRY = {
    key(r'(x_{min}, y_{min}, x_{max}, y_{max})'): sequence(text('('), sub(text('x'), 'min'), text(', '),
        sub(text('y'), 'min'), text(', '), sub(text('x'), 'max'), text(', '), sub(text('y'), 'max'), text(')')),
    key(r'(u,v)'): sequence(text('('), text('u'), text(', '), text('v'), text(')')),
    key(r'z'): Z, key(r'K'): K, key(r'T_{wc}'): TWC,
    key(r'\mathbf{p}_c=(x_c,y_c,z_c)^T'): sequence(PC, text('='), sup(sequence(text('('),
        sub(text('x'), 'c'), text(', '), sub(text('y'), 'c'), text(', '), sub(Z, 'c'), text(')')), 'T')),
    key(r'\mathbf{p}_c=zK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}'): sequence(PC, text('='), Z,
        sup(K, '−1'), matrix(text('u'), text('v'), text('1', normal=True))),
    key(r'\mathbf{p}_w=T_{wc}\begin{bmatrix}\mathbf{p}_c\\1\end{bmatrix}'): sequence(PW, text('='),
        TWC, matrix(PC, text('1', normal=True))),
    key(r'f(n)=g(n)+h(n)'): sequence(text('f'), text('('), text('n'), text(')'), text('='),
        text('g'), text('('), text('n'), text(')'), text('+'), text('h'), text('('), text('n'), text(')')),
    key(r'g(n)'): sequence(text('g'), text('('), text('n'), text(')')),
    key(r'h(n)'): sequence(text('h'), text('('), text('n'), text(')')),
}
DISPLAY_NUMBERS = {
    key(r'\mathbf{p}_c=zK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}'): '2-1',
    key(r'\mathbf{p}_w=T_{wc}\begin{bmatrix}\mathbf{p}_c\\1\end{bmatrix}'): '2-2',
    key(r'f(n)=g(n)+h(n)'): '2-3',
}
PATTERN = re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)', re.S)


def tree(latex):
    try:
        return REGISTRY[key(latex)]
    except KeyError as exc:
        raise ValueError('unreviewed manuscript formula: '+latex) from exc


def mathml_node(ast):
    kind, *args = ast
    if kind == 'text':
        value, bold, normal = args
        tag = 'mn' if value.isdigit() else 'mi' if value.isalpha() else 'mo'
        node = ET.Element(tag)
        if tag == 'mo':
            node.set('stretchy', 'false')
        if bold or normal:
            node.set('mathvariant', 'bold' if bold else 'normal')
        node.text = value
        return node
    if kind == 'seq':
        node = ET.Element('mrow')
        node.extend(mathml_node(item) for item in args[0])
    elif kind in {'sub', 'sup'}:
        node = ET.Element('msub' if kind == 'sub' else 'msup')
        node.extend(mathml_node(item) for item in args)
    elif kind == 'matrix':
        node = ET.Element('mtable')
        for item in args[0]:
            cell = ET.SubElement(ET.SubElement(node, 'mtr'), 'mtd')
            cell.append(mathml_node(item))
    elif kind == 'delim':
        node = ET.Element('mrow')
        for item in [text(args[0]), args[2], text(args[1])]:
            part = mathml_node(item)
            if part.tag == 'mo':
                part.set('stretchy', 'true')
            node.append(part)
    else:
        raise ValueError(kind)
    return node


def mathml(latex, *, display=False):
    node = ET.Element('math', xmlns='http://www.w3.org/1998/Math/MathML',
                      display='block' if display else 'inline', **{'aria-label': latex.strip()})
    node.append(mathml_node(tree(latex)))
    return ET.tostring(node, encoding='unicode')


def protect_html_math(markdown):
    replacements = {}
    def protect(match):
        display = match[1] is not None
        latex = match[1] if display else match[2]
        token = f'THESISMATHPLACEHOLDER{len(replacements):05d}TOKEN'
        rendered = mathml(latex, display=display)
        if display:
            rendered = '<span class="equation">'+rendered+'<span class="equation-number">（'+DISPLAY_NUMBERS[key(latex)]+'）</span></span>'
        replacements[token] = rendered
        return token
    return PATTERN.sub(protect, markdown), replacements


def omml_node(ast):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    kind, *args = ast
    if kind == 'text':
        value, bold, normal = args
        node = OxmlElement('m:r')
        properties = OxmlElement('m:rPr')
        style = OxmlElement('m:sty')
        style.set(qn('m:val'), 'bi' if bold else 'p' if normal or not value.isalpha() else 'i')
        properties.append(style)
        node.append(properties)
        content = OxmlElement('m:t')
        content.set(qn('xml:space'), 'preserve')
        content.text = value
        node.append(content)
        return node
    node = OxmlElement({'seq': 'm:e', 'sub': 'm:sSub', 'sup': 'm:sSup',
                       'matrix': 'm:m', 'delim': 'm:d'}[kind])
    if kind == 'seq':
        node.extend(omml_node(item) for item in args[0])
    elif kind in {'sub', 'sup'}:
        for tag, item in zip(['m:e', 'm:sub' if kind == 'sub' else 'm:sup'], args):
            part = OxmlElement(tag)
            append_omml_content(part, item)
            node.append(part)
    elif kind == 'matrix':
        for item in args[0]:
            row = OxmlElement('m:mr')
            entry = OxmlElement('m:e')
            append_omml_content(entry, item)
            row.append(entry)
            node.append(row)
    else:
        properties = OxmlElement('m:dPr')
        for tag, value in [('m:begChr', args[0]), ('m:endChr', args[1])]:
            part = OxmlElement(tag)
            part.set(qn('m:val'), value)
            properties.append(part)
        node.append(properties)
        entry = OxmlElement('m:e')
        append_omml_content(entry, args[2])
        node.append(entry)
    return node


def append_omml_content(parent, ast):
    if ast[0] == 'seq':
        parent.extend(omml_node(item) for item in ast[1])
    else:
        parent.append(omml_node(ast))


def append_word_math(paragraph, latex, *, display=False):
    from docx.oxml import OxmlElement
    equation = OxmlElement('m:oMath')
    ast = tree(latex)
    # OMML序列表达式直接置于oMath，不能增加只适用于复合结构的m:e包装。
    if ast[0] == 'seq':
        equation.extend(omml_node(item) for item in ast[1])
    else:
        equation.append(omml_node(ast))
    paragraph._p.append(equation)
    if display:
        paragraph.add_run('\t（'+DISPLAY_NUMBERS[key(latex)]+'）')


def word_inline(paragraph, line):
    previous = 0
    for match in PATTERN.finditer(line):
        paragraph.add_run(line[previous:match.start()])
        append_word_math(paragraph, match[2])
        previous = match.end()
    paragraph.add_run(line[previous:])
