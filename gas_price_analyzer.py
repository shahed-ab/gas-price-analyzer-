"""
Natural Gas Price Analyzer and Forecasting Tool
Analyzes historical natural gas prices and predicts future prices using trend analysis and seasonal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class GasPriceAnalyzer:
    """
    A class to analyze historical natural gas prices and predict future prices.
    Uses linear regression for trend analysis and seasonal decomposition for patterns.
    """
    
    def __init__(self, csv_file='Nat_Gas.csv'):
        """
        Initialize the analyzer with data from CSV file.
        
        Args:
            csv_file (str): Path to the CSV file containing dates and prices
        """
        self.df = pd.read_csv(csv_file)
        self.df.columns = ['Date', 'Price']
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate trend and seasonal factors
        self._calculate_trend()
        self._calculate_seasonal_factors()
        
    def _calculate_trend(self):
        """Calculate linear trend using least squares regression."""
        x = np.arange(len(self.df))
        y = self.df['Price'].values
        
        # Linear regression
        self.slope, self.intercept, _, _, _ = stats.linregress(x, y)
        self.df['Trend'] = self.slope * x + self.intercept
        
    def _calculate_seasonal_factors(self):
        """Calculate seasonal adjustment factors for each month."""
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Seasonal_Ratio'] = self.df['Price'] / self.df['Trend']
        
        # Average seasonal factor for each month
        self.seasonal_factors = self.df.groupby('Month')['Seasonal_Ratio'].mean().to_dict()
        
    def estimate_price(self, target_date):
        """
        Estimate the price of natural gas for a given date.
        
        Args:
            target_date (str or datetime): The date to estimate price for (format: 'YYYY-MM-DD')
            
        Returns:
            dict: Dictionary containing estimated price and components
        """
        # Convert string to datetime if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Calculate time difference from the first date
        first_date = self.df['Date'].iloc[0]
        days_diff = (target_date - first_date).days
        months_diff = days_diff / 30.44  # Average days per month
        
        # Calculate trend-based estimate
        trend_estimate = self.slope * months_diff + self.intercept
        
        # Apply seasonal adjustment
        target_month = target_date.month
        seasonal_factor = self.seasonal_factors.get(target_month, 1.0)
        
        # Final price estimate
        final_price = trend_estimate * seasonal_factor
        
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'estimated_price': round(final_price, 2),
            'trend_component': round(trend_estimate, 2),
            'seasonal_factor': round(seasonal_factor, 3),
            'is_forecast': target_date > self.df['Date'].max()
        }
    
    def generate_forecast(self, months_ahead=12):
        """
        Generate price forecast for future months.
        
        Args:
            months_ahead (int): Number of months to forecast
            
        Returns:
            pandas.DataFrame: DataFrame with forecasted dates and prices
        """
        last_date = self.df['Date'].max()
        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead + 1)]
        
        forecasts = []
        for date in forecast_dates:
            estimate = self.estimate_price(date)
            forecasts.append({
                'Date': date,
                'Forecasted_Price': estimate['estimated_price']
            })
        
        return pd.DataFrame(forecasts)
    
    def plot_analysis(self, include_forecast=True):
        """
        Create comprehensive visualization of the data and analysis.
        
        Args:
            include_forecast (bool): Whether to include 12-month forecast
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Natural Gas Price Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Historical prices with trend
        ax1 = axes[0, 0]
        ax1.plot(self.df['Date'], self.df['Price'], 'o-', label='Actual Price', linewidth=2, markersize=4)
        ax1.plot(self.df['Date'], self.df['Trend'], '--', label='Trend Line', linewidth=2, color='red', alpha=0.7)
        
        if include_forecast:
            forecast_df = self.generate_forecast()
            ax1.plot(forecast_df['Date'], forecast_df['Forecasted_Price'], 
                    's--', label='12-Month Forecast', linewidth=2, color='green', markersize=4)
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title('Price History and Forecast', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Seasonal pattern
        ax2 = axes[0, 1]
        monthly_avg = self.df.groupby('Month')['Price'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax2.plot(range(1, 13), monthly_avg.values, 'o-', linewidth=2, markersize=8, color='purple')
        ax2.set_xlabel('Month', fontsize=11)
        ax2.set_ylabel('Average Price ($)', fontsize=11)
        ax2.set_title('Seasonal Price Pattern', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Price distribution
        ax3 = axes[1, 0]
        ax3.hist(self.df['Price'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(self.df['Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${self.df["Price"].mean():.2f}')
        ax3.axvline(self.df['Price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${self.df["Price"].median():.2f}')
        ax3.set_xlabel('Price ($)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Price Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Seasonal factors
        ax4 = axes[1, 1]
        seasonal_df = pd.DataFrame(list(self.seasonal_factors.items()), columns=['Month', 'Factor'])
        seasonal_df = seasonal_df.sort_values('Month')
        colors = ['red' if f > 1 else 'green' for f in seasonal_df['Factor']]
        ax4.bar(seasonal_df['Month'], seasonal_df['Factor'], color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=1, color='black', linestyle='--', linewidth=1)
        ax4.set_xlabel('Month', fontsize=11)
        ax4.set_ylabel('Seasonal Factor', fontsize=11)
        ax4.set_title('Seasonal Adjustment Factors (>1 = Above Trend)', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(months)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('gas_price_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Analysis chart saved as 'gas_price_analysis.png'")
        plt.show()
    
    def print_summary(self):
        """Print summary statistics and insights."""
        print("\n" + "="*70)
        print("NATURAL GAS PRICE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nData Period: {self.df['Date'].min().strftime('%B %d, %Y')} to {self.df['Date'].max().strftime('%B %d, %Y')}")
        print(f"Total Data Points: {len(self.df)}")
        
        print("\n--- Price Statistics ---")
        print(f"Average Price: ${self.df['Price'].mean():.2f}")
        print(f"Median Price: ${self.df['Price'].median():.2f}")
        print(f"Minimum Price: ${self.df['Price'].min():.2f} on {self.df.loc[self.df['Price'].idxmin(), 'Date'].strftime('%Y-%m-%d')}")
        print(f"Maximum Price: ${self.df['Price'].max():.2f} on {self.df.loc[self.df['Price'].idxmax(), 'Date'].strftime('%Y-%m-%d')}")
        print(f"Standard Deviation: ${self.df['Price'].std():.2f}")
        
        print("\n--- Trend Analysis ---")
        trend_direction = "increasing" if self.slope > 0 else "decreasing"
        print(f"Overall Trend: {trend_direction.upper()}")
        print(f"Monthly Change Rate: ${abs(self.slope):.4f} per month")
        
        print("\n--- Seasonal Insights ---")
        highest_month = max(self.seasonal_factors.items(), key=lambda x: x[1])
        lowest_month = min(self.seasonal_factors.items(), key=lambda x: x[1])
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
        print(f"Highest Seasonal Demand: {months[highest_month[0]]} (Factor: {highest_month[1]:.3f})")
        print(f"Lowest Seasonal Demand: {months[lowest_month[0]]} (Factor: {lowest_month[1]:.3f})")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main function to run the analysis."""
    print("\n" + "="*70)
    print("NATURAL GAS PRICE ANALYZER AND FORECASTING TOOL")
    print("="*70)
    
    # Initialize analyzer
    try:
        analyzer = GasPriceAnalyzer('Nat_Gas.csv')
        print("✓ Data loaded successfully!")
    except FileNotFoundError:
        print("✗ Error: 'Nat_Gas.csv' file not found.")
        print("  Please ensure the CSV file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Example predictions
    print("\n" + "-"*70)
    print("EXAMPLE PRICE ESTIMATES")
    print("-"*70 + "\n")
    
    test_dates = [
        '2024-12-31',  # Future date (winter)
        '2025-06-15',  # Future date (summer)
        '2025-09-30',  # Future date (end of forecast period)
        '2023-03-15',  # Past date (interpolation)
    ]
    
    for date in test_dates:
        result = analyzer.estimate_price(date)
        forecast_label = " (FORECAST)" if result['is_forecast'] else " (HISTORICAL)"
        print(f"Date: {result['date']}{forecast_label}")
        print(f"  Estimated Price: ${result['estimated_price']}")
        print(f"  Trend Component: ${result['trend_component']}")
        print(f"  Seasonal Factor: {result['seasonal_factor']}x")
        print()
    
    # Generate and display forecast table
    print("-"*70)
    print("12-MONTH PRICE FORECAST")
    print("-"*70 + "\n")
    
    forecast_df = analyzer.generate_forecast(12)
    print(forecast_df.to_string(index=False))
    print()
    
    # Interactive mode
    print("-"*70)
    print("INTERACTIVE PRICE ESTIMATOR")
    print("-"*70)
    print("\nEnter a date to get a price estimate, or type 'plot' to see visualizations, or 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter date (YYYY-MM-DD) or command: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Gas Price Analyzer!")
            break
        elif user_input.lower() == 'plot':
            print("\nGenerating visualizations...")
            analyzer.plot_analysis(include_forecast=True)
        else:
            try:
                result = analyzer.estimate_price(user_input)
                forecast_label = "FORECAST" if result['is_forecast'] else "HISTORICAL ESTIMATE"
                print(f"\n{'='*50}")
                print(f"Price Estimate for {result['date']} ({forecast_label})")
                print(f"{'='*50}")
                print(f"Estimated Price: ${result['estimated_price']}")
                print(f"Trend Component: ${result['trend_component']}")
                print(f"Seasonal Factor: {result['seasonal_factor']}x")
                print(f"{'='*50}")
            except Exception as e:
                print(f"Error: Invalid date format. Please use YYYY-MM-DD format.")


if __name__ == "__main__":
    main()