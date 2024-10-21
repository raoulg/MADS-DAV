# Encapsulation
Encapsulation is one of the fundamental principles of object-oriented programming (OOP). It refers to the bundling of data and the methods that operate on that data within a single unit or object. This principle provides several important benefits in software design and development:

- Hiding implementation details: Encapsulation allows the internal details of an object to be hidden from the outside world. The object's internal state is kept private and can only be accessed or modified through defined methods.
- Modularity: Encapsulation helps in creating modular code. Each object is self-contained, which makes the code easier to understand, maintain, and modify.
- Flexibility and Maintainability: The internal implementation of an object can be changed without affecting other code in your project that uses the object.
- Reduced Complexity: By hiding the internal details, encapsulation reduces the complexity of the code from the perspective of external objects or functions.

# Single Responsibility Principle in Our Code Examples

The Single Responsibility Principle (SRP) is another principle of object-oriented programming and design. It states that a class should have only one reason to change, meaning it should have only one job or responsibility.

## Benefits of Adhering to SRP

By following the Single Responsibility Principle:

1. Our code becomes more modular and easier to understand
2. Each class can be tested independently
3. Changes to one aspect of the system don't affect other parts
4. The code is more flexible and easier to maintain
5. Each class can be modified or replaced without impacting others

Note how Encapsulation and SRP have overlap!

Let's look at an example to understand SRP and Encapsulation better:

```python
import pandas as pd
import re
from datetime import datetime

# Load the CSV file
df = pd.read_csv('messages.csv')

# Remove phone numbers
def remove_phone_numbers(text):
    return re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

df['message'] = df['message'].apply(remove_phone_numbers)

# Split dates from the start of messages
def split_date(row):
    match = re.match(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s(.+)', row['message'])
    if match:
        date_str, message = match.groups()
        row['date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        row['message'] = message.strip()
    return row

df = df.apply(split_date, axis=1)

# Save the processed data
df.to_csv('processed_messages.csv', index=False)

print("Data processing completed and saved to 'processed_messages.csv'")
```

There are two main things going on in this code: there is filehandling (loading and saving), and the loaded data
is preprocessed. Note how we try to make what we are doing slightly more abstract; it is 
not about just getting the job done now, but also to write code we can use more often,
in more situations.

Now, let's refactor this code to use encapsulation principles. You may want to open this file into a split-mode in your editor, such that you can study the initial code and the refactored code side by side.

We'll create four classes: one for file handling and another for preprocessing, and two dataclass object [to avoid hardcoding](never_hardcode.md).

```python
import pandas as pd
import re
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Regex:
    phone_number: str = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    date_message_split: str = r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s(.+)'

@dataclass
class Config:
    input_file: str = 'messages.csv'
    output_file: str = 'processed_messages.csv'
    message_column: str = 'message'
    date_column: str = 'date'
    timestamp_format: str = '%Y-%m-%d %H:%M:%S'

class FileHandler:
    def __init__(self, config: Config):
        self.config = config
    
    def load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.config.input_file)
    
    def save_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.config.output_file, index=False)
        print(f"Data processing completed and saved to '{self.config.output_file}'")

class DataPreprocessor:
    def __init__(self, regex: Regex, config: Config):
        self.regex = regex
        self.config = config
    
    def remove_phone_numbers(self, text: str) -> str:
        return re.sub(self.regex.phone_number, '', text)
    
    def split_date(self, row: pd.Series) -> pd.Series:
        match = re.match(self.regex.date_message_split, row[self.config.message_column])
        if match:
            date_str, message = match.groups()
            row[self.config.date_column] = datetime.strptime(date_str, self.config.timestamp_format)
            row[self.config.message_column] = message.strip()
        return row
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.config.message_column] = df[self.config.message_column].apply(self.remove_phone_numbers)
        df = df.apply(self.split_date, axis=1)
        return df

def main():
    config = Config()
    regex = Regex()
    file_handler = FileHandler(config)
    preprocessor = DataPreprocessor(regex, config)
    # Load data
    df = file_handler.load_csv()
    # Preprocess data
    df = preprocessor.preprocess(df)
    # Save processed data
    file_handler.save_csv(df)

if __name__ == "__main__":
    main()
```

We can see how the single responsibility principle and Encapsulation is applied in this refactored code:
## FileHandler Class

- **Responsibility**: Handling file I/O operations
- **Details**:
  - Deals only with loading and saving data
  - Doesn't concern itself with data content or processing
  - If file reading/writing methods need to change, only this class is affected

## DataPreprocessor Class

- **Responsibility**: Preprocessing the data
- **Details**:
  - Focuses solely on applying various preprocessing steps to the data
  - Changes to preprocessing logic are isolated to this class

## Regex Dataclass

- **Responsibility**: Storing regular expression patterns
- **Details**:
  - Acts as a simple container for regex patterns
  - Allows for easy updates or extensions to patterns without touching processing logic

## Config Dataclass

- **Responsibility**: Storing configuration settings
- **Details**:
  - Centralizes all configuration parameters
  - Enables easy updates to settings without modifying other code parts

## Practical Implications

This structure allows for easier expansion of functionality:

- To add a new preprocessing step, we only need to modify the DataPreprocessor class
- To support a new file format, we only need to update the FileHandler class

Yes, for this small script, you could argue that it is overengineerd. It also takes more lines of code. However, when your code will grow, this approach will make your code much and much easier to maintain, and it will become much easier to reuse parts of your code.

I hope you will agree that the main() method is clearer in it's function. Now imagine your codebase grows to 10.000 lines of code; at that point you will be really happy you organized your code as in the second example, and avoided code like in the first example...

# Trade off (or, when to stop)
This is always a trade off. For example, we could further improve the filehandling by abstracting away how the filetype should be handled. It really depends on the scale of your project if you want this additional layer of automation, or if you just extend your FileHandler with load_parquet and save_parquet methods... 

```python
class FileHandler:
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        if self.file_path.suffix.lower() == '.csv':
            return self._load_csv()
        elif self.file_path.suffix.lower() == '.parquet':
            return self._load_parquet()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

    def save(self, df: pd.DataFrame) -> None:
        if self.file_path.suffix.lower() == '.csv':
            self._save_csv(df)
        elif self.file_path.suffix.lower() == '.parquet':
            self._save_parquet(df)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        print(f"Data saved successfully to {self.file_path}")

    def _load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

    def _load_parquet(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)

    def _save_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.file_path, index=False)

    def _save_parquet(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.file_path, index=False)
```
