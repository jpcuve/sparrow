import {FC, forwardRef} from 'react'
import {Avatar, Box, Button, Group, MantineColor, Text, SelectItemProps, Autocomplete} from '@mantine/core'
import {state} from '../store'

const TestView:FC = () => {
  const charactersList = [
    {
      image: 'https://img.icons8.com/clouds/256/000000/futurama-bender.png',
      label: 'Bender Bending Rodríguez',
      description: 'Fascinated with cooking, though has no sense of taste',
    },

    {
      image: 'https://img.icons8.com/clouds/256/000000/futurama-mom.png',
      label: 'Carol Miller',
      description: 'One of the richest people on Earth',
    },
    {
      image: 'https://img.icons8.com/clouds/256/000000/homer-simpson.png',
      label: 'Homer Simpson',
      description: 'Overweight, lazy, and often ignorant',
    },
    {
      image: 'https://img.icons8.com/clouds/256/000000/spongebob-squarepants.png',
      label: 'Spongebob Squarepants',
      description: 'Not just a sponge',
    },
  ];

  const data = charactersList.map((item) => ({ ...item, value: item.label }));

  interface ItemProps extends SelectItemProps {
    color: MantineColor;
    description: string;
    image: string;
  }

  const AutoCompleteItem = forwardRef<HTMLDivElement, ItemProps>(
    ({ description, value, image, ...others }: ItemProps, ref) => (
      <div ref={ref} {...others}>
        <Group noWrap>
          <Avatar src={image} />
          <div>
            <Text>{value}</Text>
            <Text size="xs" color="dimmed">
              {description}
            </Text>
          </div>
        </Group>
      </div>
    )
  );
  return (
    <Box>
      <Button onClick={() => state.notifyWarning('Hello')}>Open snackbar</Button>
      <Autocomplete
        label="Choose employee of the month"
        placeholder="Pick one"
        itemComponent={AutoCompleteItem}
        data={data}
        filter={(value, item) =>
          item.value.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.description.toLowerCase().includes(value.toLowerCase().trim())
        }
      />
    </Box>
  )
}

export default TestView