data "aws_subnet_ids" "default_subnets" {
  vpc_id = aws_default_vpc.default.id
}

resource "aws_default_vpc" "default" {

}

provider "aws" {
  region  = "eu-west-1"
}

resource "aws_security_group" "am_cheap_sg" {
  name   = "am_cheap_sg"
  vpc_id = aws_default_vpc.default.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = -1
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    name = "am_cheap_sg"
  }

}

resource "aws_key_pair" "ubuntu" {
  key_name   = "ubuntu"
  public_key = file(var.PUBLIC_KEY)
}

resource "aws_spot_instance_request" "am_cheap" {
  ami                    = "ami-01530420fa8c6e1c4"
  key_name               = "ubuntu"
  instance_type          = "g4dn.xlarge"
  wait_for_fulfillment = true
  spot_type = "persistent"
  tags = {
    Name = "CheapWorker"
  }
  vpc_security_group_ids = [aws_security_group.am_cheap_sg.id]
  subnet_id              = tolist(data.aws_subnet_ids.default_subnets.ids)[0]

  connection {
    type        = "ssh"
    host        = self.public_ip
    user        = var.USER
    private_key = file(var.PRIVATE_KEY)
  }

  provisioner "remote-exec" {
    inline = [
      "git clone https://github.com/mrdvince/heth.git",
      "echo will probably do something else",
    ]
  }
}
output "aws_security_group_am_cheap_details" {
  value = aws_security_group.am_cheap_sg
}

output "am_cheap_public_dns" {
  value = aws_spot_instance_request.am_cheap.public_dns
}