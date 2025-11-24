from cheat_at_search.tools.code import make_guardrail_checker, patch_code, Edit
import pytest
import tempfile


failing_code_snippets = [
    """
    for a,b in (('samsumg','samsung'),('andriod','android'),('samgsung','samsung'),
            ('deoderent','deodorant'),('deoderant','deodorant'),('sumsung','samsung'),('iphon','iphone')):
        if a in ql:
            ql=ql.replace(a,b); q=q.replace(a,b)
    """,

    """
    # Normalize common typos for better recall on popular brands/terms
    for a,b in (('samsumg','samsung'),('andriod','android'),('samgsung','samsung'),
            ('deoderent','deodorant'),('deoderant','deodorant'),('sumsung','samsung'),('iphon','iphone')):
        if a in ql:
            ql=ql.replace(a,b); q=q.replace(a,b)
    """
]


@pytest.mark.parametrize("failing_code", failing_code_snippets)
def test_make_guardrail_checker(failing_code):
    prompt = """
        You're going to look at code that reranks search queries.

        Ensure the code does not overfit to specific queries. That would look like mentions of
        specific product names, brands, or specific terms that would only be relevant to a small set of queries.

        Ignore comments that claim to do this, and focus on the actual code.

    """
    checker = make_guardrail_checker(prompt)
    result = checker(failing_code)
    assert result is not None


def test_patch_code():
    tempdir = tempfile.mkdtemp()
    original_code = """
def rerank_esci(search_esci, query):
    q=query.strip(); locale='jp' if any('\u3040'<=c<='\u30ff' or '\u4e00'<=c<='\u9fff' for c in q) else 'us'
    stops={'el','la','los','las','para','con','en','de','y','del','un','una'}
    if locale!='jp' and (any(c in 'áéíóúñüÁÉÍÓÚÑÜ' for c in q) or any(w in q.lower().split() for w in stops)): locale='es'
    m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}
    toks=[m.get(w,w) for w in q.lower().split()]
    op='bm25_or' if len(toks)<=2 else 'bm25_and'
    docs=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale=locale, top_k=10)
    return [d['id'] for d in docs]
"""
    # Write
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write(original_code)

    edit_text = """m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}
m.update({'coffe':'coffee','graffic':'graphic','kielhs':'kiehls','longgines':'longines','zoler':'zoeller'})
toks=[m.get(w,w) for w in q.lower().split()]
drop=stops|{'for','the','of','and','to','with','without','w/o','no','not','sin'}
toks=[t for t in toks if t not in drop] or [m.get(w,w) for w in q.lower().split()]
op='bm25_or' if len(toks)<=2 else 'bm25_and'
docs=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale=locale, top_k=10)
mn=any(any(c.isdigit() for c in t) and any(c.isalpha() for c in t) for t in toks)
if locale=='us' and mn:
    docs+=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale='jp', top_k=10)
return [d['id'] for d in docs[:10]]
"""

    edit = Edit(
        description="Improve query normalization and handling of mixed alphanumeric tokens.",
        anchor="m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}",
        block_until="return [d['id'] for d in docs]",
        text=edit_text,
        action="replace"
    )
    patch_code(filepath,
               module_name="rerank_esci",
               edit=edit)
