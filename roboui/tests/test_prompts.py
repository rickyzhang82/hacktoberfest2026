from roboui.prompts import code_prompt, vision_prompt


def test_vision_prompt_variables_and_roles():
    assert set(vision_prompt.input_variables) == {"max_words", "user_notes"}
    msgs = vision_prompt.invoke({"max_words": 150, "user_notes": "dark"}).messages
    assert [m.type for m in msgs] == ["system", "human"]
    assert "150" in msgs[1].content and "dark" in msgs[1].content
    assert "Do not write code" in msgs[1].content


def test_code_prompt_enforces_stack_constraints():
    msgs = code_prompt.invoke({"description": "A login form", "user_notes": "none"}).messages
    system, human = msgs[0].content, msgs[1].content
    assert "cdn.tailwindcss.com" in system
    assert "vanilla JavaScript" in system
    assert "no network requests" in system
    assert "A login form" in human


def test_prompts_do_not_leak_braces():
    msgs = code_prompt.invoke({"description": "x", "user_notes": "y"}).messages
    assert "{description}" not in msgs[1].content
