"""
CQRS (Command Query Responsibility Segregation) Pattern Implementation

Separates read and write operations for improved scalability and performance.
"""
from .commands import *
from .queries import *
from .event_sourcing import *
from .read_models import *
