import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SmartDataAnalyzer:
    def __init__(self, data_path=None, data_frame=None):
        """Initialize the analyzer with either a file path or DataFrame"""
        self.df = None
        if data_path:
            self.load_data(data_path)
        elif data_frame is not None:
            self.df = data_frame
        self.setup_plotting_style()

    def load_data(self, data_path):
        """Load data from CSV file with UTF-8 encoding"""
        try:
            self.df = pd.read_csv(data_path, encoding='utf-8')
            print(f"Successfully loaded data from {data_path}")
            print(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def setup_plotting_style(self):
        """Configure plotting style for better aesthetics"""
        sns.set_style("darkgrid")
        sns.set_palette("deep")
        sns.set_context("talk")  # Larger fonts for better readability
        plt.rcParams['figure.figsize'] = [12, 8]  # Default larger figure size
        plt.rcParams['font.size'] = 14  # Increase font size

    def preprocess_data(self):
        """Preprocess specific columns for numerical analysis"""
        if self.df is None:
            print("No data loaded!")
            return

        # Remove ₹ and commas from price columns and convert to float
        for col in ['discounted_price', 'actual_price']:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace({'₹': '', ',': ''}, regex=True)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # Handle invalid entries as NaN

        # Remove commas from rating_count and convert to float
        if 'rating_count' in self.df.columns:
            self.df['rating_count'] = self.df['rating_count'].replace({',': ''}, regex=True)
            self.df['rating_count'] = pd.to_numeric(self.df['rating_count'], errors='coerce')

        # Remove % from discount_percentage and convert to float
        if 'discount_percentage' in self.df.columns:
            self.df['discount_percentage'] = self.df['discount_percentage'].replace({'%': ''}, regex=True)
            self.df['discount_percentage'] = pd.to_numeric(self.df['discount_percentage'], errors='coerce')

        # Clean rating column: Replace invalid entries like '|' with NaN and convert to float
        if 'rating' in self.df.columns:
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')  # Invalid becomes NaN

        print("Preprocessed price, discount, rating_count, and rating columns")

    def clean_data(self):
        """Basic data cleaning operations"""
        if self.df is None:
            print("No data loaded!")
            return

        # Remove duplicate rows
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")

        # Handle missing values
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                if self.df[column].dtype in ['int64', 'float64']:
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                else:
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        print("Missing values handled")

    def basic_statistics(self):
        """Generate basic statistical analysis with additional stats like skewness and kurtosis"""
        if self.df is None:
            print("No data loaded!")
            return None

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        stats = self.df[numeric_cols].describe().T
        stats['skew'] = self.df[numeric_cols].skew()
        stats['kurtosis'] = self.df[numeric_cols].kurt()

        categorical_stats = {}
        for column in self.df.columns:
            if self.df[column].dtype not in ['int64', 'float64']:
                categorical_stats[column] = {
                    'Unique Values': self.df[column].nunique(),
                    'Most Common': self.df[column].mode()[0],
                    'Frequency': self.df[column].value_counts().iloc[0]
                }
        categorical_df = pd.DataFrame(categorical_stats).T

        return stats, categorical_df

    def visualize_distribution(self, column, plot_type='histogram'):
        """Visualize distribution of a column with enhanced aesthetics"""
        if self.df is None or column not in self.df.columns:
            print("Invalid data or column name!")
            return

        fig = plt.figure(figsize=(12, 8))
        if plot_type == 'histogram' and self.df[column].dtype in ['int64', 'float64']:
            sns.histplot(data=self.df, x=column, kde=True, color='blue', edgecolor='black', linewidth=1.5)
            plt.title(f'Distribution of {column}', fontsize=18, pad=20)
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
        elif plot_type == 'box' and self.df[column].dtype in ['int64', 'float64']:
            sns.boxplot(y=self.df[column], color='green', fliersize=5, linewidth=1.5)
            plt.title(f'Box Plot of {column}', fontsize=18, pad=20)
            plt.ylabel(column, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
        elif plot_type == 'count':
            # Limit to top 10 categories for readability
            top_categories = self.df[column].value_counts().nlargest(10).index
            sns.countplot(data=self.df[self.df[column].isin(top_categories)], x=column, edgecolor='black', linewidth=1.5)
            plt.title(f'Top 10 Count Plot of {column}', fontsize=18, pad=20)
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Perform correlation analysis for numeric columns with enhanced heatmap"""
        if self.df is None:
            print("No data loaded!")
            return

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return

        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, linecolor='white', annot_kws={"size": 12})
        plt.title('Correlation Matrix', fontsize=18, pad=20)
        plt.tight_layout()
        plt.show()
        return correlation_matrix

    def outlier_detection(self, column, method='zscore', threshold=3):
        """Detect outliers in a numeric column with handling for NaNs"""
        if self.df is None or column not in self.df.columns:
            print("Invalid data or column name!")
            return None

        if self.df[column].dtype not in ['int64', 'float64']:
            print("Outlier detection only works for numeric columns")
            return None

        # Drop NaNs for accurate calculation
        data_clean = self.df[column].dropna()

        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean))
            outliers = self.df.loc[data_clean.index][z_scores > threshold]
        elif method == 'iqr':
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df.loc[data_clean.index][(data_clean < lower_bound) | (data_clean > upper_bound)]

        print(f"Found {len(outliers)} outliers in {column} using {method} method")
        return outliers

    def additional_visualizations(self):
        """Additional statistical visualizations"""
        if self.df is None:
            print("No data loaded!")
            return

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # Scatter Plot: Discounted Price vs Rating
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=self.df['discounted_price'], y=self.df['rating'], hue=self.df.get('category', None), palette='deep', s=100, edgecolor='black')
        plt.title('Scatter Plot of Discounted Price vs Rating', fontsize=18, pad=20)
        plt.xlabel('Discounted Price (₹)', fontsize=14)
        plt.ylabel('Rating', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Category')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Pairplot for numerical columns
        sns.pairplot(self.df[numeric_cols], diag_kind='kde', markers='+', palette='husl', height=3)
        plt.suptitle('Pairplot of Numerical Columns', y=1.02, fontsize=18)
        plt.tight_layout()
        plt.show()

        # Violin Plot for rating by category (top 5 categories)
        top_categories = self.df['category'].value_counts().nlargest(5).index
        plt.figure(figsize=(14, 8))
        sns.violinplot(x=self.df['category'][self.df['category'].isin(top_categories)], y=self.df['rating'], palette='Set3', inner='quartile', linewidth=1.5)
        plt.title('Violin Plot of Rating by Top 5 Categories', fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Rating', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        # File path set to project folder
        analyzer = SmartDataAnalyzer(data_path="data.csv")
        
        # Preprocess the data
        analyzer.preprocess_data()
        
        # Clean the data
        analyzer.clean_data()
        
        # Get basic statistics
        numeric_stats, categorical_stats = analyzer.basic_statistics()
        print("\nEnhanced Numeric Statistics Table (with Skew and Kurtosis):")
        print(numeric_stats)
        print("\nCategorical Statistics Table:")
        print(categorical_stats)
        
        # Visualize distributions
        analyzer.visualize_distribution('discounted_price', 'histogram')
        analyzer.visualize_distribution('rating', 'box')
        analyzer.visualize_distribution('category', 'count')
        
        # Correlation analysis
        analyzer.correlation_analysis()
        
        # Additional visualizations
        analyzer.additional_visualizations()
        
        # Outlier detection
        outliers = analyzer.outlier_detection('discounted_price', 'iqr')
        print("\nSample outliers:")
        print(outliers[['product_name', 'discounted_price']].head())
        
    except Exception as e:
        print(f"Error in analysis: {e}")