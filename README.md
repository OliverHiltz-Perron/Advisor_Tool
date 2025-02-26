````markdown:README.md
# Academic Advisor Assistant

An AI-powered academic advising tool that provides instant answers to questions about courses, faculty, and academic policies.

## Features

- **Course Information**: Get details about course descriptions, prerequisites, and availability
- **Faculty Profiles**: Learn about professors' research interests and teaching areas
- **Student Policies**: Access information from the student handbook and academic policies
- **User-Friendly Interface**: Simple web interface for asking questions and getting answers
- **Source Attribution**: View the sources used to generate each answer

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/academic-advisor-assistant.git
cd academic-advisor-assistant
````

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Type your question in the text box and click "Submit"

4. View the answer in the "Answer" tab and source details in the "Source Details" tab

## Project Structure

```
academic-advisor-assistant/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
└── Prompts/             # Prompt template files
    ├── router_prompt.txt
    ├── text_qa_template_student.txt
    ├── text_qa_template_prof.txt
    └── text_qa_template_courses.txt
```

## Dependencies

- llama_index - For creating and querying the knowledge base
- openai - For accessing OpenAI's language models
- gradio - For the web interface
- chromadb - For vector storage
- python-dotenv - For environment variable management
- pandas - For data manipulation

## Data Sources

The system draws information from three main sources:

1. Course descriptions and catalogs
2. Faculty profiles and research interests
3. Student handbook and academic policies

## Security Notes

- Never commit your `.env` file or expose your API keys
- The `.env` file is included in `.gitignore`
- Regularly rotate your API keys for security
- Keep your dependencies updated to patch security vulnerabilities

## Contributing

1. Fork the repository
2. Create a new branch for your feature:

```bash
git checkout -b feature/your-feature-name
```

3. Commit your changes:

```bash
git commit -m "Add your commit message"
```

4. Push to the branch:

```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**

   - Ensure your `.env` file exists and contains a valid API key
   - Check that python-dotenv is properly installed

2. **Database Connection**

   - Verify that the ChromaDB directories exist and have proper permissions
   - Check for any corrupted vector store files

3. **Memory Issues**
   - Consider reducing the `similarity_top_k` parameter if experiencing memory problems
   - Monitor RAM usage when handling large documents

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LlamaIndex](https://github.com/jerryjliu/llama_index)
- Interface created using [Gradio](https://gradio.app/)
- Vector storage powered by [Chroma](https://www.trychroma.com/)
- OpenAI for providing the language models

## Support

For technical issues or questions:

1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the development team

---

**Note**: This project is for educational purposes and should be used as a supplementary tool, not as a replacement for human academic advisors.

```

This README provides comprehensive documentation covering:
- Installation and setup
- Usage instructions
- Project structure
- Security considerations
- Troubleshooting guide
- Contribution guidelines
- Support information

The format is clean and well-organized, making it easy for users to find the information they need. The markdown formatting ensures it will render properly on GitHub or other platforms that support markdown.
```
