'''
Asks user to build financial goal profile
'''

def get_choice(prompt, options):
    """
    Prompts the user for a choice and validates the input.

    This helper function ensures that the user's input is a valid integer
    within the range of available options, re-prompting the user for valid
    input if an invalid value is entered.

    Args:
        prompt (str): The text to display to the user as a prompt.
        options (list): A list of valid options the user can choose from.

    Returns:
        The selected option from the 'options' list.
    """
        
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def getUserProfile():
    """
    Collects a user's investment profile through a series of questions.

    The function presents a number of multiple-choice questions to the user
    and uses the get_choice helper function to validate their input. The
    collected preferences are then compiled into a list.

    Returns:
        list: A list containing the user's selected preferences in a
              pre-defined order.
    """
    time_horizon_options = [1, 4, 8, 15, 25]
    desired_growth_options = [2, 5, 10, 16, 21]
    fluctuation_options = [5, 10, 15, 20, 35]
    worse_case_options = [15, 25, 35, 45, 100]
    minimum_etf_age = [10, 5, 3, 1, 0]
    risk_preference = [[3, 1], [2, 1], [1, 1], [1, 2], [1, 3]]

    print("\nHello and welcome to ETF Navigator!\nPlease answer the following questions...")

    user_time_horizon = get_choice(
        '\n1. What is your time horizon? (1/2/3/4/5):\n'
        '\t1) 0-2 years\n\t2) 3-5 years\n\t3) 6-10 years\n\t4) 11-20 years\n\t5) 20+ years\n',
        time_horizon_options)

    user_desired_growth = get_choice(
        '\n2. What are your annual growth goals? (1/2/3/4/5):\n'
        '\t1) Beat inflation (<3%)\n\t2) Modest and reliable (3-7%)\n\t3) Steady longterm (8-12%)\n'
        '\t4) Strong returns with moderate risk (13-20%)\n\t5) High growth with greater risk (>20%)\n',
        desired_growth_options)

    user_fluctuation = get_choice(
        '\n3. How much annual fluctuation is okay with you? (1/2/3/4/5):\n'
        '\t1) Not much at all (<5%)\n\t2) Small ups and downs are okay (<10%)\n'
        '\t3) Regular market swings (<15%)\n\t4) I can handle large moves if it promotes growth (<20%)\n'
        '\t5) Volatility doesn\'t bother me (>20%)\n',
        fluctuation_options)

    user_worst_case = get_choice(
        '\n4. In the worst case, what is the greatest loss you could tolerate? (1/2/3/4/5):\n'
        '\t1) Low (<15%)\n\t2) Minor (<25%)\n\t3) Moderate (<35%)\n\t4) High (<45%)\n\t5) Very high (>45%)\n',
        worse_case_options)
    
    user_minimum_efs_age = get_choice(
        '\n5. What is the minimum amount of time you would like the ETF to have existed for? (The older the ETF the more reliable the range of data) (1/2/3/4/5):\n'
        '\t1) Very Established (>10 years)\n\t2) Moderately Established (>5 years)\n\t3) Relatively New (>3 years)\n\t4) New and Emerging (>1 year)\n\t5) No Minimimum Age (All Available ETFs)\n',
        minimum_etf_age)
    
    user_risk_preference = get_choice(
        '\n6. How would you rate your preferences for risk vs return (1/2/3/4/5):\n'
        '\t1) Risk Averse (3:1)\n\t2) Risk Conscious (2:1)\n\t3) Balanced (1:1)\n\t4) Returns Prioritized (1:2)\n\t5) Return Focused (1:3)\n',
        risk_preference)

    return [user_time_horizon, user_desired_growth, user_fluctuation, user_worst_case, user_minimum_efs_age, user_risk_preference]
