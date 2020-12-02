## How to Setup Jupyter Notebook Server as a daemon service
  - [https://towshif.github.io/site/tutorials/Python/setup-Jupyter/](https://towshif.github.io/site/tutorials/Python/setup-Jupyter/)
  - [https://program-error-review.tistory.com/14](https://program-error-review.tistory.com/14)
  - [https://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95](https://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95)

#### Issues
- nginx reverse proxy setting
	> [https://jupyterhub.readthedocs.io/en/stable/reference/config-proxy.html](https://jupyterhub.readthedocs.io/en/stable/reference/config-proxy.html)
- allow remote host setting in Jupyter
	> [https://github.com/jupyterhub/jupyterhub/issues/2230](https://github.com/jupyterhub/jupyterhub/issues/2230)
- jupyter, tornado version conflict
	> [https://github.com/jupyter/notebook/issues/4439](https://github.com/jupyter/notebook/issues/4439)

<details>
  <summary>Details</summary>
  <p>

#### /etc/nginx/conf.d/jupyter.conf
```nginx.conf
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
	listen       443 ssl http2;
	listen       [::]:443 ssl http2;
	server_name  jupyter.rolroralra.com;
	#root         /usr/share/nginx/html;

	# Load configuration files for the default server block.
	#include /etc/nginx/default.d/*.conf;
	include /etc/nginx/default.d/certbot_ssl.conf;	# managed by Certbot

  add_header Strict-Transport-Security max-age=15768000;

	# Managing literal requests to the JupyterHub front end
	location / {
		proxy_pass       http://localhost:8888;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header Host $host;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

		# websocket headers
		proxy_http_version 1.1;
		proxy_set_header Upgrade $http_upgrade;
		proxy_set_header Connection $connection_upgrade;

		proxy_buffering off;
	}

 	#Managing requests to verify letsencrypt host
 	location ~ /.well-known {
 		allow all;
 	}


	error_page 404 /404.html;
		location = /40x.html {
	}

	error_page 500 502 503 504 /50x.html;
		location = /50x.html {
	}
}
```
  </p>
</details>
