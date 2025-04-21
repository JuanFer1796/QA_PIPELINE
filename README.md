

# Proyecto Final LLM

**Note:** The application interface and code comments are primarily in Spanish.

## Overview

This project showcases the development of an intelligent agent system capable of:

- **Retrieving Financial News:** Utilizing multiple agents to fetch and process financial news data.
- **Mathematical Operations:** Performing arithmetic operations such as addition and subtraction.
- **Game Development:** Programming a basic video game using Pygame.


## Features

- **Multi-Agent Architecture:** Implements several agents, each designed for specific tasks, ensuring modularity and scalability.
- **LangChain Integration:** Leverages LangChain for efficient LLM operations and prompt management.
- **Financial News Analysis:** Agents fetch and summarize current financial news, providing concise insights.
- **Arithmetic Computations:** Processes user-inputted arithmetic expressions and returns accurate results.
- **Pygame Integration:** Demonstrates the capability to generate and run a simple Pygame-based video game.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JuanFer1796/proyecto_final_LLM.git
   cd proyecto_final_LLM
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\\Scripts\\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Set Up Environment Variables:**

   Ensure you have the necessary API keys and configurations set in a `.env` file or as environment variables.

2. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

3. **Access the Interface:**

   Navigate to `http://localhost:8501` in your web browser to interact with the application.

## Demonstration

A video demonstration of the project's capabilities can be found here: [YouTube Video](https://youtu.be/33TWrjQOY7s)

## Project Structure

```
proyecto_final_LLM/
├── app.py
├── agents/
│   ├── news_agent.py
│   ├── math_agent.py
│   └── game_agent.py
├── utils/
│   └── helper_functions.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

