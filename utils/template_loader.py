def load_template(file_name):
    with open(file_name, 'r') as file:
        return file.read()

def load_templates():
    router_prompt = load_template('prompts/router_prompt.txt')
    templates = {
        'student': load_template('prompts/text_qa_template_student.txt'),
        'prof': load_template('prompts/text_qa_template_prof.txt'),
        'courses': load_template('prompts/text_qa_template_courses.txt')
    }
    return router_prompt, templates
