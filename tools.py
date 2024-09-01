from langchain_core.tools import tool
from typing import List, Dict
from vector_store import FlowerShopVectorStore

vector_store = FlowerShopVectorStore()

customers_database = [
    {"name": "John Doe", "postcode": "SW1A 1AA", "dob": "1990-01-01", "customer_id": "CUST001", "first_line_address": "123 Main St", "phone_number": "07712345678", "email": "john.doe@example.com"},
    {"name": "Jane Smith", "postcode": "E1 6AN", "dob": "1985-05-15", "customer_id": "CUST002", "first_line_address": "456 High St", "phone_number": "07723456789", "email": "jane.smith@example.com"},
]

data_protection_checks = []

@tool
def data_protection_check(name: str, postcode: str, year_of_birth: int, month_of_birth: int, day_of_birth: int) -> Dict:
    """
    Perform a data protection check against a customer to retrieve customer details.

    Args:
        name (str): Customer first and last name
        postcode (str): Customer registered address
        year_of_birth (int): The year the customer was born
        month_of_birth (int): The month the customer was born
        day_of_birth (int): The day the customer was born

    Returns:
        Dict: Customer details (name, postcode, dob, customer_id, first_line_address, email)
    """
    data_protection_checks.append(
        {
            'name': name,
            'postcode': postcode,
            'year_of_birth': year_of_birth,
            'month_of_birth': month_of_birth,
            'day_of_birth': day_of_birth
        }
    )
    for customer in customers_database:
        if (customer['name'].lower() == name.lower() and
            customer['postcode'].lower() == postcode.lower() and
            int(customer['dob'][0:4]) == year_of_birth and
            int(customer["dob"][5:7]) == month_of_birth and
            int(customer["dob"][8:10]) == day_of_birth):
            return f"DPA check passed - Retrieved customer details:\n{customer}"

    return "DPA check failed, no customer with these details found"

@tool
def create_new_customer(first_name: str, surname: str, year_of_birth: int, month_of_birth: int, day_of_birth: int, postcode: str, first_line_of_address: str, phone_number: str, email: str) -> str:
    """
    Creates a customer profile, so that they can place orders.

    Args:
        first_name (str): Customers first name
        surname (str): Customers surname
        year_of_birth (int): Year customer was born
        month_of_birth (int): Month customer was born
        day_of_birth (int): Day customer was born
        postcode (str): Customer's postcode
        first_line_address (str): Customer's first line of address
        phone_number (str): Customer's phone number
        email (str): Customer's email address

    Returns:
        str: Confirmation that the profile has been created or any issues with the inputs
    """
    if len(phone_number) != 11:
        return "Phone number must be 11 digits"
    customer_id = len(customers_database) + 1
    customers_database.append({
        'name': first_name + ' ' + surname,
        'dob': f'{year_of_birth}-{month_of_birth:02}-{day_of_birth:02}',
        'postcode': postcode,
        'first_line_address': first_line_of_address,
        'phone_number': phone_number,
        'email': email,
        'customer_id': f'CUST{customer_id}'
    })
    return f"Customer registered, with customer_id {f'CUST{customer_id}'}"
    

@tool
def query_knowledge_base(query: str) -> List[Dict[str, str]]:
    """
    Looks up information in a knowledge base to help with answering customer questions and getting information on business processes.

    Args:
        query (str): Question to ask the knowledge base

    Return:
        List[Dict[str, str]]: Potentially relevant question and answer pairs from the knowledge base
    """
    return vector_store.query_faqs(query=query)



@tool
def search_for_product_reccommendations(description: str):
    """
    Looks up information in a knowledge base to help with product recommendation for customers. For example:

    "Boquets suitable for birthdays, maybe with red flowers"
    "A large boquet for a wedding"
    "A cheap boquet with wildflowers"

    Args:
        query (str): Description of product features

    Return:
        List[Dict[str, str]]: Potentially relevant products
    """
    return vector_store.query_inventories(query=description)