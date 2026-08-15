TOOLS = [
    {
    'name': 'check_code',

    'description': (
        'Execute a Python script in an isolated sandbox to verify that it runs successfully '
        'before finalising your answer. The script is executed in a temporary working directory '
        'containing the carbon intensity dataset. The execution environment has an empty environment '
        '(env={}), so do not rely on environment variables or external state. Use this tool to detect '
        'syntax errors, runtime errors, missing imports, incorrect file paths, or other execution failures. '
        'If execution fails, inspect stderr, correct the script, and try again. Only finalise your answer'
        ' once it executes successfully.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'code': {
                'type': 'string',
                'description': 'Complete Python script to execute.'
            },
        },
        'required': ['code']
    }
}
]