package platform

default allow = false

# Allow all authenticated requests
allow if {
    input.user.authenticated == true
}

# Allow admin access to everything
allow if {
    input.user.role == "admin"
}

# Version access control
version_access if {
    input.user.role == "admin"
}

version_access if {
    input.user.role == "user"
    input.resource.version == "v2"
}

# Basic health check endpoint
allow if {
    input.path == "/health"
}
