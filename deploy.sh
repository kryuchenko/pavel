#!/bin/bash
# PAVEL - One-Click Deployment Script
# Deploys the entire PAVEL system with MongoDB and all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo -e "${BLUE}"
cat << 'EOF'
 ____   ____   _____ _      
|  _ \ / ___| |  ___| |     
| |_) |\___ \ | |_  | |     
|  __/ ___) ||  _| | |___  
|_|   |____/ |_|   |_____|
                          
Problem & Anomaly Vector Embedding Locator
One-Click Deployment Script
EOF
echo -e "${NC}"

# Check prerequisites
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed!"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running!"
        echo "Start Docker Desktop or run: sudo systemctl start docker"
        exit 1
    fi
    
    success "Docker is running"
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed!"
        echo "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    success "Docker Compose is available"
}

# Environment setup
setup_env() {
    log "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            success "Created .env from .env.example"
        else
            cat > .env << 'EOF'
# PAVEL Configuration
MONGO_PASSWORD=pavel123
PAVEL_DEFAULT_APP_ID=com.nianticlabs.pokemongo
PAVEL_LOG_LEVEL=INFO
EOF
            success "Created default .env file"
        fi
    else
        success ".env file already exists"
    fi
    
    # Create required directories
    mkdir -p logs models data secrets
    success "Created required directories"
}

# Build and start services
deploy() {
    log "Building and starting PAVEL services..."
    
    # Stop existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build and start
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d --build
    else
        docker compose up -d --build
    fi
    
    success "PAVEL services started"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for MongoDB
    log "Waiting for MongoDB..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec pavel-mongodb mongosh --quiet --eval "db.adminCommand('ping')" &>/dev/null; then
            success "MongoDB is ready"
            break
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    if [ $timeout -le 0 ]; then
        error "MongoDB failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for PAVEL app
    log "Waiting for PAVEL application..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker exec pavel-app python -c "from src.pavel.core.config import get_config; get_config()" &>/dev/null; then
            success "PAVEL application is ready"
            break
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    if [ $timeout -le 0 ]; then
        warn "PAVEL application may still be initializing"
    fi
}

# Show status and usage
show_status() {
    echo ""
    echo -e "${GREEN}🎉 PAVEL deployed successfully!${NC}"
    echo ""
    echo "📊 Services Status:"
    if command -v docker-compose &> /dev/null; then
        docker-compose ps
    else
        docker compose ps
    fi
    
    echo ""
    echo "🔗 Access Points:"
    echo "  • MongoDB: localhost:27017 (user: pavel, password: pavel123)"
    echo "  • Application: localhost:8080"
    echo ""
    echo "🚀 Usage Examples:"
    echo "  # Search reviews"
    echo "  docker exec pavel-app python search_reviews.py \"excellent game\" --limit 5"
    echo ""
    echo "  # Collect new reviews"
    echo "  docker exec pavel-app python collect_reviews.py --app-id com.example.app"
    echo ""
    echo "  # Show statistics"
    echo "  docker exec pavel-app python search_reviews.py --stats"
    echo ""
    echo "🛠️ Management:"
    echo "  # Stop all services"
    echo "  docker-compose down"
    echo ""
    echo "  # View logs"
    echo "  docker-compose logs -f pavel"
    echo ""
    echo "  # Restart services"
    echo "  docker-compose restart"
    echo ""
    echo "  # Shell access"
    echo "  docker exec -it pavel-app bash"
    echo ""
}

# Cleanup function
cleanup() {
    log "Stopping PAVEL services..."
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    else
        docker compose down
    fi
    success "Services stopped"
}

# Main execution
main() {
    case "${1:-deploy}" in
        deploy)
            log "Starting PAVEL deployment..."
            check_docker
            check_docker_compose
            setup_env
            deploy
            wait_for_services
            show_status
            ;;
        stop)
            cleanup
            ;;
        restart)
            cleanup
            sleep 2
            main deploy
            ;;
        status)
            if command -v docker-compose &> /dev/null; then
                docker-compose ps
            else
                docker compose ps
            fi
            ;;
        logs)
            if command -v docker-compose &> /dev/null; then
                docker-compose logs -f pavel
            else
                docker compose logs -f pavel
            fi
            ;;
        *)
            echo "Usage: $0 [deploy|stop|restart|status|logs]"
            echo ""
            echo "Commands:"
            echo "  deploy  - Deploy PAVEL (default)"
            echo "  stop    - Stop all services"
            echo "  restart - Restart all services"
            echo "  status  - Show service status"
            echo "  logs    - Show application logs"
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap cleanup EXIT

# Run main function
main "$@"