FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js site-config.js favicon.svg /usr/share/nginx/html/

EXPOSE 80
HEALTHCHECK --interval=20s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1/health || exit 1
